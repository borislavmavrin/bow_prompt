from evaluator import LLMEvaluator, LLMEvaluatorFS
from llm_provider import HFInf, Ollama
from learner import Learner, LearnerUCB, LearnerRidge
from vocabulary import get_flan_vocab
from tasks import get_bbh_causal_judgement, get_boolq
from pathlib import Path


SEED = 2344
EPSILON = 0.5
BEST_INSTRUCTION_LEN = 30
INSTRUCTION_LEN = 50
NUM_EVAL_EXAMPLES = 30
NUM_STEPS = 30
# model_id = "qwen2.5:0.5b"
# model_id = "llama3.2:1b"
# model_id = "qwen2.5:1.5b"
# model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
# hf_inf = HFInf(model_id, os.getenv("HF_TOKEN"), 120)
# model_id="meta-llama/Llama-3.1-8B-Instruct"
model_id = "qwen2.5:0.5b"
# task
boolq_task = get_boolq()


learner_class_map = {
    "vanila": Learner,
    "ucb": LearnerUCB,
    "ridge": LearnerRidge
}
for modle_id in ["qwen2.5:0.5b", "llama3.2:1b", "qwen2.5:1.5b"]:
    for learner_class in learner_class_map.keys():
        results_path = f"results/{learner_class}__{model_id.replace('/', '_')}"
        assert not Path(results_path).exists()

        ollama = Ollama(model_id)
        evaluator = LLMEvaluator(provider=ollama, tasks=boolq_task[2:], seed=SEED, num_examples=NUM_EVAL_EXAMPLES)
        learner = learner_class_map[learner_class](evaluator=evaluator, word_list=get_flan_vocab(), topk=BEST_INSTRUCTION_LEN, results_path=results_path)
        learner.run(steps=NUM_STEPS, num_idxs_to_choose=INSTRUCTION_LEN, epsilon=EPSILON)
        best_score, best_instruction = learner.evaluate_best_idxs()
        print(best_score, best_instruction)
