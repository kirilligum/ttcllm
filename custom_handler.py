import asyncio
import os
import litellm
from litellm.llms.custom_llm import CustomLLM


class MyCustomLLM(CustomLLM):
    def completion(self, *args, **kwargs) -> litellm.ModelResponse:
        messages = kwargs.pop(
            "messages", None
        )  # Remove "messages" from kwargs and store it
        kwargs.pop("model", None)  # Remove the incoming model value
        return litellm.completion(
            model="llama-3.1-8b-instant",  # Override model changed to llama-3.1-8b-instant as required
            messages=messages,
            base_url=os.environ["GROQ_API_BASE"],
            api_key=os.environ["GROQ_API_KEY"],
        )  # type: ignore

    async def acompletion(self, *args, **kwargs) -> litellm.ModelResponse:
        return await asyncio.to_thread(self.completion, *args, **kwargs)


my_custom_llm = MyCustomLLM()
