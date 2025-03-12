import re
from typing import List
import litellm
from litellm import CustomLLM, completion

class IterativeReasoningLLM(CustomLLM):
    def completion(self, model: str, messages: List[dict], **kwargs) -> litellm.ModelResponse:
        # Extract N from model name (e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct-wait-6")
        match = re.search(r'-wait-(\d+)$', model)
        n_iterations = min(int(match.group(1)) if match else 0, 20)

        total_prompt_tokens = 0
        total_completion_tokens = 0

        # Create a local copy of the original conversation
        conversation_history = messages.copy()
        assistant_messages = []

        # Perform (N+1) calls: one initial call + N iterative refinement steps.
        for i in range(n_iterations + 1):
            response = completion(
                # Use the base model (DeepInfra settings defined in config will be used)
                model="openai/meta-llama/Meta-Llama-3.1-8B-Instruct",
                messages=conversation_history,
                **kwargs
            )

            usage = response.usage
            total_prompt_tokens += usage["prompt_tokens"]
            total_completion_tokens += usage["completion_tokens"]

            # Record the assistant's answer
            assistant_reply = response.choices[0].message.content
            assistant_messages.append(assistant_reply)

            # Append the assistant answer to conversation history
            conversation_history.append({"role": "assistant", "content": assistant_reply})

            # If not on the final iteration, append the fixed user prompt for refinement
            if i < n_iterations:
                conversation_history.append({"role": "user", "content": "wait, check your reasoning"})

        # Separate all but the final response for the reasoning block.
        if len(assistant_messages) > 1:
            test_time_compute_block = "\n".join(assistant_messages[:-1])
            final_message = assistant_messages[-1]
            combined_message = f"<test-time-compute>\n{test_time_compute_block}\n</test-time-compute>\n\n{final_message}"
        else:
            combined_message = assistant_messages[0]

        return litellm.ModelResponse(
            id=response.id,
            object="chat.completion",
            created=response.created,
            model=model,
            choices=[{"message": {"role": "assistant", "content": combined_message},
                      "finish_reason": "stop"}],
            usage={
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens
            }
        )

# Instantiate and register the custom LLM handler.
iterative_llm = IterativeReasoningLLM()
litellm.custom_provider_map = [
    {"provider": "iterative-llm", "custom_handler": iterative_llm}
]
