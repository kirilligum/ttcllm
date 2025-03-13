import asyncio
import litellm
from litellm.llms.custom_llm import CustomLLM


class MyCustomLLM(CustomLLM):
    def completion(self, *args, **kwargs) -> litellm.ModelResponse:
        messages = kwargs.pop(
            "messages", None
        )  # Remove "messages" from kwargs and store it
        kwargs.pop("model", None)  # Remove the incoming model value
        return litellm.completion(
            model="gpt-4o-mini",  # Override model to gpt-4o-mini as required
            messages=messages,
        )  # type: ignore

    async def acompletion(self, *args, **kwargs) -> litellm.ModelResponse:
        return await asyncio.to_thread(self.completion, *args, **kwargs)


my_custom_llm = MyCustomLLM()
