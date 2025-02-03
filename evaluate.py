import os
import json
from tqdm import tqdm
import random


class Evaluator:
    def __init__(self, provider):
        self.max_tokens = 32
        self._llm_call = provider.llm_call
        self.tasks = json.load(open("data/causal_judgement.json"))["examples"]
        random.seed(3424)
        random.shuffle(self.tasks)
    
    def evaluate(self, instruction: str, num_examples=10, remove_emb_instruction=True):
        score = 0
        responses = list()
        for task in tqdm(self.tasks[:num_examples]):
            tast_input = task["input"]
            if remove_emb_instruction:
                tast_input = "\n".join(task["input"].split("\n")[1:])

            messages = [dict(role="user", content="\n\n".join([instruction, tast_input]))]
            response = self._llm_call(messages, self.max_tokens)
            if task["target"].lower() in response.replace(",", " ").replace(".", " ").strip().lower().split():
                score += 1
            responses.append(response)
        return score / len(self.tasks[:num_examples]), responses
