from evaluator import Evaluator
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from tqdm import tqdm
from pathlib import Path


class Learner:
    def __init__(self, evaluator: Evaluator, word_list, verbose=True, topk=30, results_path=None):
        self.evaluator = evaluator
        self.words = list()
        self.scores = list()
        self.sample_space = np.arange(len(word_list))
        self.thetas_hat = np.zeros(len(word_list))
        self.word_list = word_list
        self.verbose = verbose
        self.topk = topk
        self.results_path = Path(results_path) if results_path is not None else None
    
    def add(self, words, score):
        self.words.append(words)
        self.scores.append(score)
    
    def choose_idxs(self, num_idxs, method='uniform'):
        if method == 'uniform':
            return np.random.choice(self.sample_space, num_idxs, replace=False)
        elif method == 'proportional':
            p = np.zeros(len(self.sample_space))
            non_nan_idx = np.argwhere(~np.isnan(self.thetas_hat)).flatten()   
            pos_idx = (self.thetas_hat > 0.).flatten()
            pos_non_nan_idx = non_nan_idx  # np.array([i for i in non_nan_idx if i in pos_idx])
            p[pos_non_nan_idx] = self.thetas_hat[pos_non_nan_idx]
            if p.min() < 0:
                p += np.abs(p.min())
            assert p.sum() >= 0
            if p.sum() > 0:
                p /= p.sum()
            elif p.sum() == 0.:
                p = np.ones(len(self.sample_space))
                p /= p.sum()
            return np.random.choice(self.sample_space, min(num_idxs, p[p>0].size), p=p, replace=False)
        else:
            raise ValueError('wrong sampling method')

    def vectorize(self, words):
        X = list()
        for ws in words:
            x = np.zeros(len(self.word_list))
            idx = np.array([self.word_list.index(w) for w in ws])
            x[idx] = 1.
            X.append(x)
        return np.stack(X)
    
    def update_weights(self):
        X = self.vectorize(self.words)
        y = np.array(self.scores)
        X = sm.add_constant(X)
        model = sm.OLS(y, X)
        results = model.fit()
        thetas_hat = results.params[-len(self.word_list):]
        non_nan_idx = np.argwhere(~np.isnan(thetas_hat)).flatten()
        if non_nan_idx.size != 0:
            if self.verbose:
                print("updating weights")
            self.thetas_hat[non_nan_idx] = thetas_hat[non_nan_idx]

    def run(self, steps, num_idxs_to_choose, epsilon=0.1):
        for step in tqdm(range(steps), disable=not self.verbose):
            if np.random.random() < epsilon:
                idxs = self.choose_idxs(num_idxs_to_choose, method='uniform')
            else:
                idxs = self.choose_idxs(num_idxs_to_choose, method='proportional')
            words = [self.word_list[i] for i in idxs]
            instruction = " ".join(words)
            score = self.evaluator.evaluate(instruction)
            
            # track best score
            best_score, best_instruction = self.evaluate_best_idxs()
            num_pos_weights = np.sum(self.thetas_hat > 0.)
            num_nan_weights = np.argwhere(np.isnan(self.thetas_hat)).flatten().size
            if self.verbose:
                print(f"\nstep: {step}| best score: {best_score} | num pos weights: {num_pos_weights} | num neg weights: {np.sum(self.thetas_hat <= 0.)} | num nan weights: {num_nan_weights}")
                print(f"\n{len(best_instruction.split())} | {best_instruction}")
            self.add(words, score)
            self.update_weights()

            self.save_results(best_score, best_instruction)
    
    def evaluate_best_idxs(self):
        range_idxs = np.arange(len(self.sample_space))
        # assert best_idxs.size == self.thetas_hat.size
        pos_thetas_idx = range_idxs[(self.thetas_hat > 0.).flatten()]
        sorted_thetas_idx = np.argsort(-self.thetas_hat).flatten()
        best_idxs = np.array([i for i in sorted_thetas_idx if i in pos_thetas_idx])
        best_idxs_len = best_idxs.size
        best_idxs = best_idxs[:self.topk]
        if self.verbose:
            print(f"\n{pos_thetas_idx.size} and {sorted_thetas_idx.size}, {best_idxs_len} -> {best_idxs.size}")
        instruction = " ".join([self.word_list[i] for i in best_idxs])
        max_reward = self.evaluator.evaluate(instruction)
        return max_reward, instruction
    
    def save_results(self, best_score, best_instruction):
        if self.results_path is not None:
            self.results_path.mkdir(parents=True, exist_ok=True)
            with open(self.results_path / "best_scores.csv", "a") as f:
                f.write(f"{best_score},{best_instruction}\n")


class LearnerUCB(Learner):
    def update_weights(self):
        X = self.vectorize(self.words)
        y = np.array(self.scores)
        X = sm.add_constant(X)
        model = sm.OLS(y, X)
        results = model.fit()
        # update only non nans
        thetas_hat = results.conf_int()[-len(self.word_list):, 1]
        non_nan_idx = np.argwhere(~np.isnan(thetas_hat)).flatten()
        if non_nan_idx.size != 0:
            if self.verbose:
                print("updating weights")
            self.thetas_hat[non_nan_idx] = thetas_hat[non_nan_idx]
