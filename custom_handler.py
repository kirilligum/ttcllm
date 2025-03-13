import litellm
from litellm import CustomLLM, completion, get_llm_provider


class MyCustomLLM(CustomLLM):
    def completion(self, *args, **kwargs) -> litellm.ModelResponse:
        return litellm.completion(
            # model="gpt-4o-mini",
            model="groq/llama-3.3-70b-specdec",
            # model="g33",
            messages=[{"role": "user", "content": "Hello world"}],
            # mock_response="Hi!",
        )  # type: ignore

    async def acompletion(self, *args, **kwargs) -> litellm.ModelResponse:
        return litellm.completion(
            # model="g33",
            model="groq/llama-3.3-70b-specdec",
            # model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello world"}],
            # mock_response="Hi!",
        )  # type: ignore


my_custom_llm = MyCustomLLM()
