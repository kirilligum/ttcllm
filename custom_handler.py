import asyncio
import os
import re
import litellm
from typing import List
from litellm.llms.custom_llm import CustomLLM


class MyCustomLLM(CustomLLM):
    def completion(self, model: str, messages: List[dict], **kwargs) -> litellm.ModelResponse:
        # Extract N from model name (e.g., "my-custom-llm/llama-3.1-8b-instant-wait-6")
        match = re.search(r'-wait-(\d+)$', model)
        n_iterations = min(int(match.group(1)) if match else 0, 20)
        
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        # Create a local copy of the original conversation
        conversation_history = messages.copy()
        assistant_messages = []
        
        # Determine the base model (remove "-wait-<n>" if present)
        base_model = model[:match.start()] if match else model
        
        # Perform (n_iterations + 1) calls: one initial call plus iterative refinement steps.
        for i in range(n_iterations + 1):
            response = litellm.completion(
                model=base_model,
                messages=conversation_history,
                base_url=os.environ["GROQ_API_BASE"],
                api_key=os.environ["GROQ_API_KEY"],
                **kwargs
            )  # type: ignore
            
            usage = response.usage
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)
            
            # Extract assistant reply from the response
            assistant_reply = response.choices[0].message.content
            assistant_messages.append(assistant_reply)
            
            # Append the full assistant reply to the conversation history
            conversation_history.append({"role": "assistant", "content": assistant_reply})
            
            # If not on the final iteration, ask the model to re-examine its reasoning
            if i < n_iterations:
                conversation_history.append({"role": "user", "content": "wait, check your reasoning"})
        
        # Build the combined answer by wrapping all but the final response
        if len(assistant_messages) > 1:
            test_time_compute_block = "\n".join(assistant_messages[:-1])
            final_message = assistant_messages[-1]
            combined_message = (
                f"<test-time-compute>\n{test_time_compute_block}\n</test-time-compute>\n\n{final_message}"
            )
        else:
            combined_message = assistant_messages[0]
        
        # Return a new ModelResponse containing the combined answer and aggregated token usage
        return litellm.ModelResponse(
            id=response.id,
            object="chat.completion",
            created=response.created,
            model=model,
            choices=[
                {
                    "message": {"role": "assistant", "content": combined_message},
                    "finish_reason": "stop"
                }
            ],
            usage={
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens
            }
        )

    async def acompletion(self, *args, **kwargs) -> litellm.ModelResponse:
        return await asyncio.to_thread(self.completion, *args, **kwargs)


my_custom_llm = MyCustomLLM()
