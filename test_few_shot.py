from llm_provider import HFInf, Ollama
from dotenv import load_dotenv
from evaluator import LLMEvaluator, LLMEvaluatorFS
from huggingface_hub import login
import os
from tasks import get_boolq


SEED = 2344
# load_dotenv()
# login(os.getenv("HF_TOKEN"))
model_id="qwen2.5:1.5b"
# model_id = "microsoft/Phi-3.5-mini-instruct"
# hf_inf = HFInf(model_id, os.getenv("HF_TOKEN"), 120)
ollama = Ollama(model_id=model_id)
boolq_task = get_boolq()
evaluator = LLMEvaluatorFS(provider=ollama, tasks=boolq_task[2:], few_shot_tasks=boolq_task[:2], seed=SEED)
score, responses = evaluator.evaluate("")
print(score)
print(responses)
