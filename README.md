# TTC-LLM (Test-Time Compute LLM Proxy)

A FastAPI-based, OpenAI API–compatible proxy server that enhances DeepInfra-hosted LLM responses with iterative reasoning.

## Features

- **N-step reasoning:** Dynamically repeats model calls with `"wait, check your reasoning"` for improved results.
- **Token usage tracking:** Aggregates token counts across iterations and returns a usage summary.
- **OpenAI API compatible:** Works as a drop-in replacement – simply change your model name to include a `-wait-N` suffix (maximum N=20).

## Folder Structure

```
ttc-llm/
├── src/
│   ├── server.py            # Main LiteLLM proxy server
│   ├── custom_handler.py    # Custom LL model handler (with iterative reasoning)
│   ├── utils.py             # Helper functions
│   └── __init__.py          # (empty)
├── config/
│   └── config.yaml          # Configuration file
├── README.md                # Documentation
├── requirements.txt         # Dependencies
├── .gitignore               # Ignored files
└── LICENSE                  # MIT License
```

## Installation

1. Clone the repository and change directory:
   ```bash
   git clone https://github.com/yourgithub/ttc-llm.git
   cd ttc-llm
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Edit `config/config.yaml` to supply your DeepInfra API key and modify any other settings as needed.

## Running the Server

Start the proxy server with:
```bash
python src/server.py
```

By default the server will run on `http://0.0.0.0:4000`.

## Usage Example

Using OpenAI's Python SDK:
```python
import openai

openai.api_base = "http://localhost:4000/v1"
openai.api_key = "sk-ProxyKey"  # A dummy key for your proxy

response = openai.ChatCompletion.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-wait-6",
    messages=[{"role": "user", "content": "What is 15!?"}]
)

print(response["choices"][0]["message"]["content"])
```

### API Response Format

The final response includes a reasoning block enclosed in `<test-time-compute>` tags and the final answer outside that block. For example:
```json
{
  "id": "chatcmpl-<id>",
  "object": "chat.completion",
  "created": 1721955063,
  "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-wait-6",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "<test-time-compute>\n...initial and intermediate answers...\n</test-time-compute>\n\nFinal answer."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 200,
    "total_tokens": 300
  }
}
```

Happy computing!
