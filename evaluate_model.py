import os, time
import json
from vllm import LLM, SamplingParams
from datasets import load_from_disk, load_dataset
from utils import DATASET_KEYS, RESPONSE_EXTRACTOR, RESPONSE_COMPARATOR
import pandas as pd
import argparse
import numpy as np
import re
from typing import List, Union, Optional
import random

# -------------------------------------------
LETTERS = list("ABCDEFGHIJ")              # keep in sync with your prompt!

_punct_pat = re.compile(r"[^A-Za-z0-9]+")

def normalise_pred(
    pred: Union[str, int, float],
    options: List[str],
) -> Optional[int]:
    """
    Convert model output to a **0‑based** index in ``options``.

    The function is quite permissive: besides numeric or single‑letter
    answers ("2", "B"), it now also recognises *substring* matches such as
    ``"I think the answer is Paris"`` when one of the choices is
    ``"Paris"``.

    Parameters
    ----------
    pred     : str | int | float
        Raw model output. May include explanation or extra tokens.
    options  : list[str]
        The option strings, in the same order used in the prompt.

    Returns
    -------
    int | None
        The 0‑based index of the chosen option, or *None* if no match
        could be established.
    """
    # 1. If the model already returned a valid integer, honour it
    if isinstance(pred, (int, float)):
        idx = int(pred)
        if 0 <= idx < len(options):
            return idx

    # 2. Normalise the textual output
    text = str(pred).strip()
    cleaned = _punct_pat.sub("", text).upper()  # only alphanumerics

    # 2a. Single‑letter answer?  e.g. "C" / "(b)" / "Answer: D"
    if len(cleaned) == 1 and cleaned in LETTERS[: len(options)]:
        return LETTERS.index(cleaned)


    try:

        # 2b. Bare index?  e.g. "2"
        if cleaned.isdigit():
            idx = int(cleaned)
            if 0 <= idx < len(options):
                return idx

        # 3. Exact case‑insensitive match against choices
        text_lower = text.lower()
        for i, opt in enumerate(options):
            opt_lower = opt.lower()
            if text_lower == opt_lower:
                return i

        # 3b. Substring match – covers outputs like "My answer is Paris"
        for i, opt in enumerate(options):
            if opt.lower() in text_lower:
                return i

    except:
        pass  # silently ignore any exception

    # 4. OPTIONAL: add fuzzy‑matching (rapidfuzz) if desired

    # 5. No match could be found
    return 0


# utils/dataset_loaders/gpqa.py
from datasets import load_from_disk



CHOICES = ["A", "B", "C", "D"]

def load_gpqa(split: str = "train"):
    ds = load_from_disk("datasets/gpqa")

    def _wrap(ex):
        # Create a list of all options with their labels
        options = [
            ("A", ex["Correct Answer"]),
            ("B", ex["Incorrect Answer 1"]),
            ("C", ex["Incorrect Answer 2"]),
            ("D", ex["Incorrect Answer 3"]),
        ]
        # Shuffle the options
        random.shuffle(options)
        # Find the index of the correct answer after shuffling
        for idx, (label, text) in enumerate(options):
            if text == ex["Correct Answer"]:
                correct_idx = idx
                break
        # Build the prompt with shuffled options
        #body = "\n".join(f"({CHOICES[i]}) {opt[1]}" for i, opt in enumerate(options))
        #prompt = f"{ex['Question']}\n\n{body}\n\nAnswer:"
        #opts = example["options"]          # list[str] (len ≤ 10)
        prompt = (
            f"{ex['Question']}\n\n"
            #"Choose the correct answer by writing only the corresponding letter (A–D).\n\n"
            + "\n".join(f"({CHOICES[i]}) {opt[1]}" for i, opt in enumerate(options))
            + "\n\nAnswer:"
        )
        return {"question": prompt, "solution": correct_idx}

    ds['test'] = ds['train']
    return ds.map(_wrap)

# MMLU‑Pro has (up to) 10 choices, labelled A‑J
LETTERS = list("ABCDEFGHIJ")

def load_mmlu_pro(split: str = "test") -> "datasets.Dataset":
    """
    Return the requested split of the TIGER‑Lab/MMLU‑Pro benchmark with
    exactly two columns:
        • question : the full multiple‑choice prompt
        • answer   : the zero‑based index of the correct choice
    Parameters
    ----------
    split : {"validation", "test"}   (default "test")
    """
    # 1. pull the dataset dict from HF
    ds: DatasetDict = load_dataset("TIGER-Lab/MMLU-Pro")  # splits: validation, test

    # 2. build prompt/label pair for each row
    def _wrap(example):
        opts = example["options"]          # list[str] (len ≤ 10)
        prompt = (
            f"{example['question']}\n\n"
            #"Choose the correct answer by writing only the corresponding letter (A–J).\n\n"
            + "\n".join(f"({l}) {o}" for l, o in zip(LETTERS, opts))
            + "\n\nAnswer:"
        )
        return {
            "question": prompt,
            "answer":   int(example["answer_index"]),      # already 0‑based
        }

    # 3. column names to drop (identical in every split)
    COLS_TO_REMOVE = [
        "question_id", "question", "answer",
        "answer_index", "cot_content", "category", "src",
    ]

    # 4. apply to every split, then hand back the requested one
    wrapped = ds.map(_wrap, remove_columns=COLS_TO_REMOVE, batched=False)
    return wrapped

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--dataset', type=str)
parser.add_argument('--scale', type=str, default='1.5B')
parser.add_argument('--tok_limit', type=int, default=32768)
args = parser.parse_args()
os.environ['TOKENIZERS_PARALLELISM'] = "false"

dataset_name = args.dataset
model_path = args.model_path
scale = args.scale
tok_limit = args.tok_limit
dataset_name = args.dataset
results = {}

print("Dataset:", dataset_name, "\nScale:", scale)

QUESTION_KEY = DATASET_KEYS[dataset_name]["question"]
ANSWER_KEY = DATASET_KEYS[dataset_name]["answer"]
eq = RESPONSE_COMPARATOR[dataset_name]

if dataset_name == 'datasets/converted_aime_dataset':
    dataset = load_from_disk(dataset_name)
    TEST_N = 1
    MAX_TOKENS = tok_limit
    TEST_TEMPERATURE = 0
    MAX_TEST_SAMPLES = 300
elif dataset_name == 'di-zhang-fdu/MATH500':
    dataset = load_dataset(dataset_name)
    TEST_N = 1
    MAX_TOKENS = tok_limit
    TEST_TEMPERATURE = 0
    MAX_TEST_SAMPLES = 1000
elif dataset_name == 'opencompass/AIME2025':
    dataset = load_dataset(dataset_name, 'AIME2025-II')
    TEST_N = 1
    MAX_TOKENS = tok_limit
    TEST_TEMPERATURE = 0
    MAX_TEST_SAMPLES = 15
elif dataset_name == 'TIGER-Lab/MMLU-Pro':
    dataset = load_mmlu_pro()
    TEST_N = 1
    MAX_TOKENS = tok_limit
    TEST_TEMPERATURE = 0
    MAX_TEST_SAMPLES = 1000
elif dataset_name == 'Idavidrein/gpqa':
    dataset = load_gpqa()
    TEST_N = 1
    MAX_TOKENS = tok_limit
    TEST_TEMPERATURE = 0
    MAX_TEST_SAMPLES = 450 #448
elif dataset_name == 'isaiahbjork/cot-logic-reasoning':
    dataset = load_dataset(dataset_name)
    dataset['test'] = dataset['train']
    TEST_N             = 1
    MAX_TOKENS         = tok_limit
    TEST_TEMPERATURE   = 0
    MAX_TEST_SAMPLES   = 1000
elif dataset_name == 'openai/gsm8k':
    dataset = load_dataset(dataset_name, 'main')
    TEST_N = 1
    MAX_TOKENS = tok_limit
    TEST_TEMPERATURE = 0
    MAX_TEST_SAMPLES = 1319
elif dataset_name == 'TIGER-Lab/TheoremQA':
    dataset = load_dataset(dataset_name)
    TEST_N = 1
    MAX_TOKENS = tok_limit
    TEST_TEMPERATURE = 0
    MAX_TEST_SAMPLES = 800

per_level_length = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
per_level_acc = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}

def get_scores(ds, outputs, save_file_name=None):
    predictions, golds = [], []
    results = []
    for input, output in zip(ds, outputs):
        gold = RESPONSE_EXTRACTOR[dataset_name](str(input[ANSWER_KEY]))
        if 'Theorem' in dataset_name:
            #print("CORRECT")
            gold = RESPONSE_EXTRACTOR['TheoremQAFalse'](str(input[ANSWER_KEY]))
            #print("GOLD", gold)
        prediction = [
            RESPONSE_EXTRACTOR[dataset_name](resp.text)
            for resp in output.outputs
        ]
        if 'mmlu' in dataset_name or 'MMLU' in dataset_name:
            prediction = [normalise_pred(predd, input["options"]) for predd in prediction]
        if 'gpqa' in dataset_name:
            optionss = [
                input["Correct Answer"],
                input["Incorrect Answer 1"],
                input["Incorrect Answer 2"],
                input["Incorrect Answer 3"],
            ]
            #print("??", optionss)
            prediction = [normalise_pred(predd, optionss) for predd in prediction]
        #print("AAA", gold, prediction)
        predictions.append(prediction)
        golds.append(gold)
        #print("QUEST", input[QUESTION_KEY])
        #print("OUTPUT", [resp.text for resp in output.outputs])

        accuracy_res = []

        for pred in prediction:
            score = eq(gold, pred)
            if isinstance(score, bool):
                accuracy_res.append(score)
            else:
                accuracy_res.append(float(score))

        if "MATH500" in dataset_name:

            #print("SHOWING FOR", input['level'])
            #print("QUESTION", input[QUESTION_KEY])
            #print("ANSWER", gold)
            #print("NUMBER OF TOKENS", (sum([len(resp.token_ids) for resp in output.outputs]) / len(output.outputs)))
            #print("MODEL", output.outputs[0].text)

            per_level_length[input['level']].append(sum([len(resp.token_ids) for resp in output.outputs]) / len(output.outputs))
            if accuracy_res[0]:
                per_level_acc[input['level']].append(1)
            else:
                per_level_acc[input['level']].append(0)
            #print(input['level'], accuracy_res, sum([len(resp.token_ids) for resp in output.outputs]) / len(output.outputs))

        results.append(
            {
                QUESTION_KEY: input[QUESTION_KEY],
                ANSWER_KEY: str(input[ANSWER_KEY]),
                "responses": [resp.text for resp in output.outputs],
                "prediction": prediction,
                "gold": gold,
                "tokens": sum([len(resp.token_ids) for resp in output.outputs]) / len(output.outputs),
                "accuracy": accuracy_res,
            }
        )
    if save_file_name is not None:
        with open(save_file_name, 'w') as f:
            json.dump(results, f, indent=4)

    results = pd.DataFrame(results)
    predictions, golds, tokens = results["prediction"], results["gold"], results["tokens"]
    if 'cot-logic' in dataset_name:
        pass_at_1 = sum([eq(g, p[0]) for p, g in zip(predictions, golds)]) / len(predictions)
    else:
        pass_at_1 = sum([any([eq(g, pred) for pred in p[:1]]) for p, g in zip(predictions, golds)]) / len(predictions)
    pass_at_k_list = []
    acc_at_k_list = []
    k = TEST_N
    print("Average tokens:", sum(tokens) / len(tokens))
    for i in range(k):
        if 'cot-logic' in dataset_name:
            #print("COT LOGIC")
            pass_at_i = sum([
                sum([eq(g, pred) for pred in p[:i+1]]) / (i + 1)
                for p, g in zip(predictions, golds)
            ]) / len(predictions)
        else:
            pass_at_i = sum([any([eq(g, pred) for pred in p[:i+1]]) for p, g in zip(predictions, golds)]) / len(predictions)
        #pass_at_i = sum([any([eq(g, pred) for pred in p[:i+1]]) for p, g in zip(predictions, golds)]) / len(predictions)
        acc_at_i = sum([eq(g, p[i]) for p, g in zip(predictions, golds)]) / len(predictions)
        acc_at_k_list.append(acc_at_i)
        pass_at_k_list.append(pass_at_i)
        print(
            f"Pass @ {i+1}: {pass_at_i}"
        )

    def get_most_common(solns):
        soln_counts = {}
        for soln in solns:
            if soln is None:
                continue
            added = False
            for other_solns in solns:
                if eq(soln, other_solns):
                    added = True
                    soln_counts[soln] = soln_counts.get(soln, 0) + 1
            if not added:
                soln_counts[soln] = 1
        if len(soln_counts) == 0:
            return None
        return max(soln_counts, key=soln_counts.get)
    
    predictions_maj = [get_most_common(p) for p in predictions]
    all_preds = sum([[eq(golds[i], p) for p in predictions[i]] for i in range(len(predictions))], [])
    avg_pass_rate = sum(all_preds) / len(all_preds)
    pass_at_n = sum([eq(g, p) for p, g in zip(predictions_maj, golds)]) / len(predictions)
    print(
        f"Pass @ 1(with majority): {pass_at_n}"
    )
    
    return {
        'pass@1': pass_at_1,
        'pass@1(majority)': sum([eq(g, p) for p, g in zip(predictions_maj, golds)]) / len(predictions),
        'average_pass_rate': avg_pass_rate,
        'std_pass_rate': np.std(acc_at_k_list),
        'acc@k': acc_at_k_list,
        'pass@k': pass_at_k_list,
        'avg_tokens': sum(tokens) / len(tokens)
    }


def evaluate_model(model_name):
    test_prompts = []
    model = LLM(model_name, tokenizer=f'deepseek-ai/DeepSeek-R1-Distill-Qwen-{scale}', gpu_memory_utilization=0.9, tensor_parallel_size=1)    
    test_ds = dataset['test'].shuffle(seed=0).select(range(min(MAX_TEST_SAMPLES, len(dataset['test']))))
    
    for x in test_ds:
        if dataset_name == 'isaiahbjork/cot-logic-reasoning':
            user_content = x[QUESTION_KEY]            # ask as‑is
        else:
            user_content = (
                f"Please reason step by step, and put your final answer "
                f"within \\boxed{{}}. Question: {x[QUESTION_KEY]}"
            )
        prompt = [{"role": "user", "content": user_content}]
        #print("GOING TO MODEL", prompt)
        prompt_tokens = model.llm_engine.tokenizer.tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
        test_prompts.append(prompt_tokens)
    
    sampling_params = SamplingParams(
        temperature=TEST_TEMPERATURE,
        max_tokens=MAX_TOKENS,
        n=TEST_N
    )
    sampling_params.stop_token_ids = [model.llm_engine.tokenizer.tokenizer.eos_token_id]
    print("Generating test outputs...")
    #print(model.llm_engine.tokenizer.tokenizer.decode(test_prompts[0], skip_special_tokens=False))
    start_time = time.time()
    test_outputs = model.generate(prompt_token_ids=test_prompts, sampling_params=sampling_params, use_tqdm=True)
    #print("OUT OF THE MODEL", test_outputs)
    end_time = time.time()
    test_scores = get_scores(test_ds, test_outputs, f"outputs/{dataset_name.replace('/', '_')}_results_{model_path.replace('/', '_')}_{tok_limit}.json")
    print("Test:", test_scores)
    time_taken = end_time - start_time
    print("Time taken:", time_taken)

    #for level in range(6):
    #    print("Level", level, "->")
    #    if len(per_level_length[level]) == 0:
    #        print("EMPTY")
    #        continue
    #    print("Average length:", sum(per_level_length[level]) / len(per_level_length[level]))
    #    print("Average accuracy:", sum(per_level_acc[level]) / len(per_level_acc[level]))

    return {'test': test_scores, 'time_taken': time_taken}

print("Found model_path:", model_path)
print("This is not a checkpoint, will evaluate directly...")
scores = evaluate_model(model_path)
results[model_path] = scores

with open(f'results/{dataset_name.replace("/", "_")}_results_{model_path.replace("/", "_")}_{tok_limit}.json', 'w') as f:
    json.dump(results, f, indent=4)
