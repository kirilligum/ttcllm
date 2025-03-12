import re

def extract_n_from_model(model_name: str) -> int:
    """
    Extracts the iteration count N from a model name like:
    'meta-llama/Meta-Llama-3.1-8B-Instruct-wait-6'.
    Caps N to 20.
    """
    match = re.search(r'-wait-(\d+)$', model_name)
    return min(int(match.group(1)) if match else 0, 20)
