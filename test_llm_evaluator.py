from dotenv import load_dotenv
from evaluator import LLMEvaluator
from huggingface_hub import login
import os
from llm_provider import HFInf, Ollama
from learner import Learner, LearnerUCB
from vocabulary import get_flan_vocab
from tasks import get_bbh_causal_judgement, get_boolq


NUM_TRIALS = 30
NUM_WORDS = 100
NUM_EXAMPLES = 50
# load_dotenv()
# login(os.getenv("HF_TOKEN"))
# model_id="meta-llama/Llama-3.1-8B-Instruct"
# model_id = "llama3.2:1b"
model_id = "qwen2.5:0.5b"
# model_id = "qwen2.5:1.5b"
# hf_inf = HFInf(model_id, os.getenv("HF_TOKEN"), 120)
ollama = Ollama(model_id)
evaluator = LLMEvaluator(provider=ollama, tasks=get_boolq())
learner = Learner(evaluator=evaluator, word_list=get_flan_vocab(), topk=30, results_path=f"results/{model_id}")
learner.run(steps=300, num_idxs_to_choose=50, epsilon=0.5)
best_score, best_instruction = learner.evaluate_best_idxs()
print(best_score, best_instruction)
