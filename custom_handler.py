import asyncio
import os
import re
import litellm
from litellm.llms.custom_llm import CustomLLM


class MyCustomLLM(CustomLLM):
    def completion(self, *args, **kwargs) -> litellm.ModelResponse:
        messages = kwargs.pop("messages", None)  # Remove "messages" from kwargs
        incoming_model = kwargs.pop("model", "llama-3.1-8b-instant")
        pattern = r"-wait-(\d+)$"
        match = re.search(pattern, incoming_model)
        if match:
            n_steps = min(int(match.group(1)), 20)
            base_model = incoming_model[: match.start()]
            conversation = messages.copy() if messages else []
            for _ in range(n_steps):
                response = litellm.completion(
                    model=base_model,
                    messages=conversation,
                    base_url=os.environ["GROQ_API_BASE"],
                    api_key=os.environ["GROQ_API_KEY"],
                )  # type: ignore
                conversation.append({"role": "assistant", "content": response})
                conversation.append({"role": "user", "content": "wait, check your reasoning"})
            conversation.append({"role": "user", "content": "final answer:"})
            final_response = litellm.completion(
                model=base_model,
                messages=conversation,
                base_url=os.environ["GROQ_API_BASE"],
                api_key=os.environ["GROQ_API_KEY"],
            )  # type: ignore
            return final_response
        else:
            return litellm.completion(
                model=incoming_model,
                messages=messages,
                base_url=os.environ["GROQ_API_BASE"],
                api_key=os.environ["GROQ_API_KEY"],
            )  # type: ignore

    async def acompletion(self, *args, **kwargs) -> litellm.ModelResponse:
        return await asyncio.to_thread(self.completion, *args, **kwargs)


my_custom_llm = MyCustomLLM()
