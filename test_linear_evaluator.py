from evaluator import LinearEvaluator
from learner import Learner, LearnerUCB
import numpy as np
from pathlib import Path
from tqdm import tqdm


NUM_TRIALS = 10
NUM_WORDS = 20
NUM_STEPS = 1_000
NUM_IDXS_TO_CHOOSE = NUM_WORDS // 4
METHOD = 'uniform'
RESULTS_PATH = Path(f"NUM_WORDS_{NUM_WORDS}__NUM_STEPS_{NUM_STEPS}__METHOD_{METHOD}.csv")

simple_lst = list()
for trial in range(NUM_TRIALS):
    linear_evaluator = LinearEvaluator(num_words=NUM_WORDS, verbose=False)
    learner = Learner(linear_evaluator, word_list=[str(i) for i in range(NUM_WORDS)], verbose=False)
    learner.run(steps=NUM_STEPS, method=METHOD, num_idxs_to_choose=NUM_IDXS_TO_CHOOSE)
    score, _ = learner.evaluate_best_idxs()
    simple_lst.append(score)

ucb_lst = list()
for trial in range(NUM_TRIALS):
    linear_evaluator = LinearEvaluator(num_words=NUM_WORDS, verbose=False)
    learner = LearnerUCB(linear_evaluator, word_list=[str(i) for i in range(NUM_WORDS)], verbose=False)
    learner.run(steps=NUM_STEPS, method=METHOD, num_idxs_to_choose=NUM_IDXS_TO_CHOOSE)
    score, _ = learner.evaluate_best_idxs()
    ucb_lst.append(score)
    simple_mean = np.mean(simple_lst)
ucb_mean = np.mean(ucb_lst)
open(RESULTS_PATH, "a").write(f"{NUM_IDXS_TO_CHOOSE},{float(simple_mean)},{float(ucb_mean)}\n")
print(f"simple_lst: {np.mean(simple_lst)}, +/- {np.std(simple_lst)}")
print(f"ucb_lst: {np.mean(ucb_lst)}, +/- {np.std(ucb_lst)}")

