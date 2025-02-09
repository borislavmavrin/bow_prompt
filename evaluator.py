import os
import json
from tqdm import tqdm
import random
import numpy as np
from abc import ABC, abstractmethod
from few_shot import create_few_shot_examples


class Evaluator(ABC):
    @classmethod
    @abstractmethod
    def evaluate(instruction: str) -> tuple[float, list[str]]:
        pass


class LinearEvaluator(Evaluator):
    def __init__(self, num_words, verbose=True):
        thetas = np.random.uniform(-100, 100, num_words)
        pos_idx = (thetas >= 0.).flatten()
        neg_idx = (thetas < 0.).flatten()
        thetas[pos_idx] = thetas[pos_idx] / thetas[pos_idx].sum()
        thetas[neg_idx] = thetas[neg_idx] / np.abs(thetas[neg_idx]).sum()
        assert np.isclose(thetas[pos_idx].sum(), 1.)
        assert np.isclose(thetas.sum(), 0.)
        self.thetas = thetas
        self.verbose = verbose

    def evaluate(self, instruction):
        if instruction != "":
            idxs = np.array([int(word) for word in instruction.split(" ")])
            return max(float(self.thetas[idxs].sum()), 0.)
        return 0., [""]


class LLMEvaluator(Evaluator):
    def __init__(self, provider, tasks, verbose=True, seed=None):
        self.max_tokens = 32
        self._llm_call = provider.llm_call
        self.tasks = tasks
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.tasks)
        self.verbose = verbose
    
    def evaluate(self, instruction: str, num_examples=30, remove_emb_instruction=True):
        scores = list()
        responses = list()
        for task in tqdm(self.tasks[:num_examples], disable=not self.verbose):
            tast_input = task["input"]
            messages = [dict(role="system", content=instruction), dict(role="user", content=tast_input)]
            # messages = [dict(role="user", content="\n\n".join([instruction, tast_input]))]
            response = self._llm_call(messages, self.max_tokens)
            if task["target"].lower() in response.replace(",", " ").replace(".", " ").strip().lower().split():
                scores.append(1.)
            else:
                scores.append(0.)
            responses.append(response)
        return float(np.mean(scores)), responses


class LLMEvaluatorFS(Evaluator):
    def __init__(self, provider, tasks, few_shot_tasks, verbose=True, seed=None):
        self.max_tokens = 32
        self._llm_call = provider.llm_call
        self.tasks = tasks
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.tasks)
        self.verbose = verbose
        self.few_shot_examples = create_few_shot_examples(few_shot_tasks)
    
    def evaluate(self, instruction: str, num_examples=30, remove_emb_instruction=True):
        scores = list()
        responses = list()
        for task in tqdm(self.tasks[:num_examples], disable=not self.verbose):
            messages = self.few_shot_examples
            messages.extend([dict(role="system", content=instruction), dict(role="user", content=task["input"])])
            response = self._llm_call(messages, self.max_tokens)
            if task["target"].lower() in response.replace(",", " ").replace(".", " ").strip().lower().split():
                scores.append(1.)
            else:
                scores.append(0.)
            responses.append(response)
        return float(np.mean(scores)), responses
