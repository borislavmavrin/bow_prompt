from dotenv import load_dotenv
from evaluate import Evaluator
from huggingface_hub import login
import os
import random
import json
from llm_provider import HFInf, Ollama


NUM_TRIALS = 30
NUM_WORDS = 100
NUM_EXAMPLES = 50
load_dotenv()
login(os.getenv("HF_TOKEN"))
model_id="meta-llama/Llama-3.1-8B-Instruct"
hf_inf = HFInf(model_id, os.getenv("HF_TOKEN"), 120)
ollama = Ollama(model_id="llama3.2")
evaluator = Evaluator(ollama)
eval_ins = "such subject away summer painting class customer account step to help explain role we plan natural agree"
# score, responses = evaluator.evaluate(instruction=eval_ins, num_examples=NUM_EXAMPLES)
# print(score)
# print(responses)
 
instruction_words = [w.strip() for w in open("data/1100_en_words.txt").readlines()]
ins_score = list()
for trial in range(NUM_TRIALS):
    # random.seed(123423)
    instruction = random.sample(instruction_words, k=NUM_WORDS)
    instruction = " ".join(instruction)
    score, _ = evaluator.evaluate(instruction, num_examples=NUM_EXAMPLES)
    ins_score.append(dict(score=score, instruction=instruction))
    print(score, instruction)

    json.dump(ins_score, open("data/results.json", "w"), indent=4)
