import socket
import json
from datasets import load_dataset
import requests
import time

def load_math_dataset():
    """Loads the GSM8K dataset."""
    dataset = load_dataset("hendrycks/competition_math", 'main', split="train")
    return {entry['problem']: entry['solution'] for entry in dataset}

def load_gsm8k_dataset():
    """Loads the GSM8K dataset."""
    dataset = load_dataset("gsm8k", 'main', split="train")
    return {entry['question']: entry['answer'] for entry in dataset}

def send_query_to_server(query, host='localhost', port=12345):
    """Sends a query to the server and returns the response."""
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))

        client_socket.sendall(json.dumps(query).encode('utf-8'))
        response = client_socket.recv(1024).decode('utf-8')

        return json.loads(response)
    except Exception as e:
        return {"error": str(e)}
    finally:
        client_socket.close()
    
def request_api_wrapper(url, data, try_max_times=5):
    """Synchronous request API wrapper."""
    headers = {
        "Content-Type": "application/json",
    }
    for _ in range(try_max_times):
        try:
            response = requests.post(url=url, json=data, headers=headers, timeout=180)
            response.raise_for_status()
            response = response.json()

            if "error" in response:
                raise Exception(response["error"])

            return response
        except requests.RequestException as e:
            print(f"Request error, please check: {e}")
        except Exception as e:
            print(f"Unexpected error, please check: {e}")
        time.sleep(1)

    raise Exception(f"Request error for {try_max_times} times, returning None. Please check the API server.")

if __name__ == "__main__":
    # Example queries
    # gsm8k_entries = load_gsm8k_dataset()
    # math_entries = load_math_dataset()
    # q1 = list(gsm8k_entries.keys())[0]
    # q2 = list(math_entries.keys())[0]
    # print(q1, gsm8k_entries[q1])
    # print(q2, math_entries[q2])
    query = json.load(open('error.json'))
    # query['steps'] = 120
    queries = [
        # {'query': [
        #     {'response': 'the answer is most probably \\boxed{72.0}.<|eot_id|>', 'aux_info': {'question': q1, 'answer': gsm8k_entries[q1], 'dataset_name': 'openai/gsm8k'}},
        #     {'response': 'the answer is most probably \\boxed{72.0} but i don\'t know.', 'aux_info': {'question': q1, 'answer': gsm8k_entries[q1], 'dataset_name': 'openai/gsm8k'}},
        #     {'response': 'the answer is most probably \\boxed{72.0} but i don\'t know what do you think.', 'aux_info': {'question': q1, 'answer': gsm8k_entries[q1], 'dataset_name': 'openai/gsm8k'}},
        #     # {'response': 'the answer is \\bo', 'aux_info': {'problem': q2, 'solution': math_entries[q2], 'dataset_name': 'hendrycks/competition_math'}},
        #     # {'response': 'the answer is \\bo', 'aux_info': {'problem': q2, 'solution': math_entries[q2], 'dataset_name': 'hendrycks/competition_math'}}
        # ]}
        # json.load(open('error3.log'))
        query
    ]
    url = "http://127.0.0.1:12310/query"

    for i, query in enumerate(queries):
        print(f"Sending Query {i + 1}: {query}")
        # response = send_query_to_server(query)
        response = request_api_wrapper(url, query)
        print(f"Response {i + 1}: {response}\n")
