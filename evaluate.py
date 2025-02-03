from huggingface_hub import InferenceClient
import os
import json
from tqdm import tqdm
import random


class Evaluator:
    def __init__(self, token, model_id=""):
        self.model_id = model_id,
        timeout = 120
        self.max_tokens = 32
        self.client = InferenceClient(model_id, token=token, timeout=timeout)
        self.tasks = json.load(open("data/causal_judgement.json"))["examples"]
        random.seed(3424)
        random.shuffle(self.tasks)

    def reponse(self, messages):
        output = self.client.chat.completions.create(
            messages, max_tokens=self.max_tokens
        )

        response = output.choices[0].message.content
        last_input_token_count = output.usage.prompt_tokens
        last_output_token_count = output.usage.completion_tokens
        return response
    
    def evaluate(self, instruction: str, num_examples=10, remove_emb_instruction=True):
        score = 0
        responses = list()
        for task in tqdm(self.tasks[:num_examples]):
            tast_input = task["input"]
            if remove_emb_instruction:
                tast_input = "\n".join(task["input"].split("\n")[1:])

            messages = [dict(role="user", content="\n\n".join([instruction, tast_input]))]
            response = self.reponse(messages)
            if task["target"].lower() in response.replace(",", " ").replace(".", " ").strip().lower().split():
                score += 1
            responses.append(response)
        return score / len(self.tasks[:num_examples]), responses
