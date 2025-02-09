from huggingface_hub import InferenceClient
from abc import ABC, abstractmethod
from ollama import chat
from ollama import ChatResponse
from dotenv import load_dotenv
from huggingface_hub import login
import os


class LLMProvider(ABC):
    @classmethod
    @abstractmethod
    def llm_call(self):
        pass


class HFInf(LLMProvider):
    def __init__(self, model_id, timeout):
        load_dotenv()
        login(os.getenv("HF_TOKEN"))
        self._client = InferenceClient(model_id, token=token, timeout=timeout)
    
    def llm_call(self, messages, max_tokens):
        output = self._client.chat.completions.create(
            messages, max_tokens=max_tokens
        )

        response = output.choices[0].message.content
        last_input_token_count = output.usage.prompt_tokens
        last_output_token_count = output.usage.completion_tokens
        return response


class Ollama(LLMProvider):
    def __init__(self, model_id):
        self.model_id = model_id

    def llm_call(self, messages, max_tokens):
        response = chat(model=self.model_id, messages=messages, options={"temperature": 0., "num_predict": max_tokens})
        return response.message.content
