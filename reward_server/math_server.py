'''
This file is the server for the reward function. It listens for incoming connections from the client and sends the reward to the OpenRLHF environment.
'''

import json
from flask import Flask, request, jsonify
from datasets import load_from_disk
import argparse
import numpy as np
from utils import DATASET_KEYS, RESPONSE_COMPARATOR, RESPONSE_EXTRACTOR
from transformers import AutoTokenizer
from openai import OpenAI
import math

app = Flask(__name__)


def load_dataset_dicts(datasets):
    # Load datasets into memory
    dataset_dict = {}
    print(f"Loading {datasets}...")
    for dataset_name in datasets:
        dataset = load_from_disk(dataset_name)
        if 'train' in dataset:
            dataset = dataset['train']
        print(f"Picking {dataset_name} consisting of {len(dataset)} examples.")
        question_key = DATASET_KEYS[dataset_name]['question']
        answer_key = DATASET_KEYS[dataset_name]['answer']
        for entry in dataset:
            dataset_dict[entry[question_key]] = entry[answer_key]
        
    return dataset_dict

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cosine_fn(L_gen: int, L_max: int, r0: float, rL: float) -> float:
    """Cosine interpolation between r0 and rL over token length L_gen ∈ [0, L_max]."""
    L_gen = max(0, min(L_gen, L_max))  # clamp to [0, L_max]
    t = L_gen / L_max
    return (r0 + rL) / 2 + (r0 - rL) / 2 * np.cos(np.pi * t)

def cosine_reward(correct: bool, L_gen: int) -> float:
    """Implements the piecewise-cosine shaping reward."""
    L_max = 14336
    rc0, rcL = 2.0, 1.0
    rw0, rwL = -10.0, 0.0
    r_exceed = -10.0

    if L_gen >= L_max:
        return r_exceed

    if correct:
        return cosine_fn(L_gen, L_max, rc0, rcL)
    else:
        return cosine_fn(L_gen, L_max, rw0, rwL)

@app.route('/query', methods=['POST'])
def query():
    # Main entry point for the server
    client = app.config['openai']
    try:
        metrics = {'rewards': [], 'GPT_Score3': []}
        if app.config['reward_type'] == 'GPT_Separated':
            metrics = metrics = {'rewards': [], 'GPT_Score3': []}
        for dataset_name in app.config['dataset_names']:
            metrics[f"{dataset_name}_accuracy"] = []
            metrics[f"{dataset_name}_response_length"] = []
            metrics[f"is_{dataset_name}"] = []

        tokenizer = app.config['tokenizer']
        query_dict = request.get_json()

        # Compute length of only the correct responses grouped by 'question'
        avg_length = {}
        avg_length_of_batch = []
        idxx = 0

        for query in query_dict.get('query', []):
            aux_info = query.get('aux_info', None)
            curr_dataset_name = aux_info.get('dataset_name', None)
            question = aux_info[DATASET_KEYS[curr_dataset_name]['question']]
            if question in avg_length:
                continue
            all_responses = aux_info['all_responses']
            answer = app.config['dataset_dict'].get(aux_info[DATASET_KEYS[curr_dataset_name]['question']], None)
            question_decoded = tokenizer.decode(tokenizer.encode(question, add_special_tokens=False), skip_special_tokens=True)

            idxx += 1
            for response in all_responses:
                response_len = len(tokenizer.encode(tokenizer.decode(tokenizer.encode(response, add_special_tokens=False), skip_special_tokens=True)))

                extracted_response = RESPONSE_EXTRACTOR[curr_dataset_name](response)
                extracted_answer = RESPONSE_EXTRACTOR[curr_dataset_name](answer)
                contains_eos = True
                if tokenizer.eos_token_id not in tokenizer.encode(response, add_special_tokens=False):
                    contains_eos = False
                if not contains_eos and app.config['check_eos']:
                    accuracy = 0
                else:
                    accuracy = float(RESPONSE_COMPARATOR[curr_dataset_name](extracted_response, extracted_answer))
                
                if accuracy > 0:
                    # Collect only if acc > 0
                    if question not in avg_length:
                        avg_length[question] = []
                    avg_length[question].append(response_len)
                    avg_length_of_batch.append(response_len)

        for query in query_dict.get('query', []):
            aux_info = query.get('aux_info', None)
            curr_dataset_name = aux_info.get('dataset_name', None)
            response = query.get('response', None)
            question = aux_info[DATASET_KEYS[curr_dataset_name]['question']]
            contains_eos = True
            if tokenizer.eos_token_id not in tokenizer.encode(response, add_special_tokens=False):
                contains_eos = False
            if not contains_eos and app.config['check_eos']:
                accuracy = 0
                reward = 0
                response_len = 0
            else:
                response_len = len(tokenizer.encode(tokenizer.decode(tokenizer.encode(response, add_special_tokens=False), skip_special_tokens=True))) # needed because there are special tokens in the response
                answer = app.config['dataset_dict'].get(aux_info[DATASET_KEYS[curr_dataset_name]['question']], None)
                extracted_response = RESPONSE_EXTRACTOR[curr_dataset_name](response)
                extracted_answer = RESPONSE_EXTRACTOR[curr_dataset_name](answer)
                accuracy = float(RESPONSE_COMPARATOR[curr_dataset_name](extracted_response, extracted_answer))

            if app.config['reward_type'] == 'sigmoid':
                if accuracy > 0:
                    lens = avg_length[aux_info[DATASET_KEYS[curr_dataset_name]['question']]]
                    relative_length = (response_len - np.mean(lens)) / (np.std(lens) + 1e-7) # Reward only when answer is correct.
                    reward = accuracy * (1 - app.config['alpha'] * (sigmoid(relative_length)))
                else:
                    reward = 0.0

                response_decoded = tokenizer.decode(tokenizer.encode(response, add_special_tokens=False), skip_special_tokens=True)
                after_question = response_decoded.split(question_decoded, 1)[-1].split('<｜Assistant｜>', 1)[-1]

                #print("##", after_question)

                to_GPT = f"""You are an expert evaluator tasked with scoring the conciseness of a reasoning trace from an AI model.

Conciseness means providing clear, precise, and direct reasoning.  
- High-scoring reasoning (8–10) is brief yet explicitly demonstrates the logical steps or thought processes clearly.
- Medium-scoring reasoning (5–7) might have minor redundancy, slight verbosity, or slightly unclear phrasing.
- Low-scoring reasoning (1–4) is either overly verbose, repetitive, vague, contains placeholders, or is too brief (such as immediately stating the final answer without any intermediate reasoning steps).

Do NOT reward extremely short traces that only state the final answer without reasoning.
Tags like <think>, </think>, <answer>, </answer> are acceptable and should NOT affect scoring.

Evaluate ONLY conciseness. Ignore correctness or accuracy entirely.

Provide ONLY a single integer from 1 (least concise) to 10 (most concise). Do NOT include explanations or additional text.

Reasoning Trace:
{after_question}

Conciseness Score (1-10):
"""
                
                response = client.responses.create(
                    #model="gpt-4.1-nano-2025-04-14",
                    #model="gpt-4o-mini-2024-07-18",
                    model="gpt-4.1-mini-2025-04-14",
                    input=to_GPT,
                )

                try:
                    GPT_Scoree = int(response.output_text)/10.0
                except ValueError:
                    print(f"Warning: output_text is not an integer: {response.output_text}")
                    GPT_Scoree = 5
                
                metrics['GPT_Score3'].append(GPT_Scoree)

            if app.config['reward_type'] == 'cosine':
                correct = accuracy > 0
                reward = cosine_reward(correct, response_len)

            if app.config['reward_type'] == 'GPT_Score3':
                
                if accuracy > 0:

                    response_decoded = tokenizer.decode(tokenizer.encode(response, add_special_tokens=False), skip_special_tokens=True)
                    after_question = response_decoded.split(question_decoded, 1)[-1].split('<｜Assistant｜>', 1)[-1]

                    to_GPT = f"""You are an expert evaluator tasked with scoring the conciseness of a reasoning trace from an AI model.

Conciseness means providing clear, precise, and direct reasoning.  
- High-scoring reasoning (8–10) is brief yet explicitly demonstrates the logical steps or thought processes clearly.
- Medium-scoring reasoning (5–7) might have minor redundancy, slight verbosity, or slightly unclear phrasing.
- Low-scoring reasoning (1–4) is either overly verbose, repetitive, vague, contains placeholders, or is too brief (such as immediately stating the final answer without any intermediate reasoning steps).

Do NOT reward extremely short traces that only state the final answer without reasoning.
Tags like <think>, </think>, <answer>, </answer> are acceptable and should NOT affect scoring.

Evaluate ONLY conciseness. Ignore correctness or accuracy entirely.

Provide ONLY a single integer from 1 (least concise) to 10 (most concise). Do NOT include explanations or additional text.

Reasoning Trace:
{after_question}

Conciseness Score (1-10):
"""
                    
                    response = client.responses.create(
                        #model="gpt-4.1-nano-2025-04-14",
                        #model="gpt-4o-mini-2024-07-18",
                        model="gpt-4.1-mini-2025-04-14",
                        input=to_GPT,
                    )

                    try:
                        reward = int(response.output_text)/10.0
                    except ValueError:
                        print(f"Warning: output_text is not an integer: {response.output_text}")
                        reward = 5

                else:
                    reward = 0.0

            if app.config['reward_type'] == 'GPT_Separated':

                response_decoded = tokenizer.decode(tokenizer.encode(response, add_special_tokens=False), skip_special_tokens=True)
                after_question = response_decoded.split(question_decoded, 1)[-1].split('<｜Assistant｜>', 1)[-1]

                to_GPT = f"""You are an expert evaluator tasked with scoring the conciseness of a reasoning trace from an AI model.

Conciseness means providing clear, precise, and direct reasoning.  
- High-scoring reasoning (8–10) is brief yet explicitly demonstrates the logical steps or thought processes clearly.
- Medium-scoring reasoning (5–7) might have minor redundancy, slight verbosity, or slightly unclear phrasing.
- Low-scoring reasoning (1–4) is either overly verbose, repetitive, vague, contains placeholders, or is too brief (such as immediately stating the final answer without any intermediate reasoning steps).

Do NOT reward extremely short traces that only state the final answer without reasoning.
Tags like <think>, </think>, <answer>, </answer> are acceptable and should NOT affect scoring.

Evaluate ONLY conciseness. Ignore correctness or accuracy entirely.

Provide ONLY a single integer from 1 (least concise) to 10 (most concise). Do NOT include explanations or additional text.

Reasoning Trace:
{after_question}

Conciseness Score (1-10):
"""
                    
                response = client.responses.create(
                    #model="gpt-4.1-nano-2025-04-14",
                    #model="gpt-4o-mini-2024-07-18",
                    model="",
                    input=to_GPT,
                )

                try:
                    reward = int(response.output_text)/10.0
                except ValueError:
                    print(f"Warning: output_text is not an integer: {response.output_text}")
                    reward = 5

                metrics['GPT_Score3'].append(reward)
                reward += accuracy

            metrics['rewards'].append(reward) # score can be something else as well, not just correctness
                                                
            for dataset_name in app.config['dataset_names']:
                if dataset_name == curr_dataset_name:
                    metrics[f"is_{dataset_name}"].append(1.0)
                    metrics[f"{dataset_name}_accuracy"].append(accuracy)
                    metrics[f"{dataset_name}_response_length"].append(response_len)
                else:
                    metrics[f"is_{dataset_name}"].append(float('nan'))
                    metrics[f"{dataset_name}_accuracy"].append(float('nan'))
                    metrics[f"{dataset_name}_response_length"].append(float('nan'))
    
        return jsonify(metrics), 200
    
    except Exception as e:
        # Save the dict for debugging purposes.
        print("Query:", json.dump(query_dict, open('error.json', 'w'), indent=4))
        print(str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--openai_key', type=str, default='xxx')
    parser.add_argument('--model_api', type=str, default='gpt-4.1-mini-2025-04-14')
    parser.add_argument('--address', type=str, default='0.0.0.0:100')
    parser.add_argument('--dataset_names', type=str, default='openai/gsm8k')
    parser.add_argument('--tokenizer', type=str, default='meta-llama/Llama-3.2-1B-Instruct')
    parser.add_argument('--reward_type', type=str, default='linear') # can be linear or sigmoid
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--check_eos', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    dataset_names = args.dataset_names.split(',')
    app.config['dataset_names'] = dataset_names
    
    dataset_dict = load_dataset_dicts(dataset_names)
    print(f"Server will start running on port: {args.address}. Use the URI 'http://{args.address}/query' to send queries.")
    app.config['dataset_dict'] = dataset_dict
    app.config['tokenizer'] = AutoTokenizer.from_pretrained(args.tokenizer)
    app.config['reward_type'] = args.reward_type
    app.config['alpha'] = args.alpha
    app.config['check_eos'] = args.check_eos
    app.config['openai'] = OpenAI(api_key=args.openai_key)
    
    app.run(host=args.address.split(":")[0], port=int(args.address.split(":")[1]))