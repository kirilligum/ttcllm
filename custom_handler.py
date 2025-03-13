import asyncio
import os
import re
from typing import List

import litellm
from litellm.llms.custom_llm import CustomLLM


class MyCustomLLM(CustomLLM):
    def completion(
        self, model: str, messages: List[dict], **kwargs
    ) -> litellm.ModelResponse:
        # Extract N from model name
        match = re.search(r"-wait-(\d+)$", model)
        n_iterations = min(int(match.group(1)) if match else 0, 20)

        total_prompt_tokens = 0
        total_completion_tokens = 0

        # Create a local copy of the original conversation
        conversation_history = messages.copy()
        assistant_messages = []

        # Determine the base model (remove "-wait-<n>" if present)
        base_model = model[: match.start()] if match else model

        # Perform n_iterations iterative refinement steps.
        for i in range(n_iterations):
            new_kwargs = kwargs.copy()
            new_kwargs.pop("api_key", None)
            new_kwargs.pop("api_base", None)
            response = litellm.completion(
                model=base_model,
                messages=conversation_history,
                api_base=os.environ["GROQ_API_BASE"],
                api_key=os.environ["GROQ_API_KEY"],
            )  # type: ignore

            usage = response.usage
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)

            # Extract assistant reply from the response
            assistant_reply = response.choices[0].message.content
            assistant_messages.append(assistant_reply)

            # Append the assistant reply to conversation history and prompt for further reasoning
            conversation_history.append(
                {"role": "assistant", "content": assistant_reply}
            )
            conversation_history.append(
                {"role": "user", "content": "wait, check your reasoning"}
            )

        # Append the final prompt to trigger the final answer.
        conversation_history.append({"role": "user", "content": "final answer:"})

        new_kwargs = kwargs.copy()
        new_kwargs.pop("api_key", None)
        new_kwargs.pop("api_base", None)
        final_response = litellm.completion(
            model=base_model,
            messages=conversation_history,
            api_base=os.environ["GROQ_API_BASE"],
            api_key=os.environ["GROQ_API_KEY"],
        )  # type: ignore

        usage = final_response.usage
        total_prompt_tokens += usage.get("prompt_tokens", 0)
        total_completion_tokens += usage.get("completion_tokens", 0)

        final_message = final_response.choices[0].message.content
        assistant_messages.append(final_message)

        # Build the combined answer: include prior responses as a test-time compute block and append the final answer.
        if len(assistant_messages) > 1:
            test_time_compute_block = "\n".join(assistant_messages[:-1])
            combined_message = f"<test-time-compute>\n{test_time_compute_block}\n</test-time-compute>\n\n{final_message}"
        else:
            combined_message = final_message

        # Return a new ModelResponse containing the combined answer and aggregated token usage
        return litellm.ModelResponse(
            id=response.id,
            object="chat.completion",
            created=response.created,
            model=model,
            choices=[
                {
                    "message": {"role": "assistant", "content": combined_message},
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
            },
        )

    async def acompletion(self, *args, **kwargs) -> litellm.ModelResponse:
        return await asyncio.to_thread(self.completion, *args, **kwargs)


my_custom_llm = MyCustomLLM()
