from huggingface_hub import InferenceClient


class HFInf():
    def __init__(self, model_id, token, timeout):
        self._client = InferenceClient(model_id, token=token, timeout=timeout)
    
    def llm_call(self, messages, max_tokens):
        output = self._client.chat.completions.create(
            messages, max_tokens=max_tokens
        )

        response = output.choices[0].message.content
        last_input_token_count = output.usage.prompt_tokens
        last_output_token_count = output.usage.completion_tokens
        return response
