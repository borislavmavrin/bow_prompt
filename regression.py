import json
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge


instruction_words = list(set([w.strip().lower() for w in open("data/1100_en_words.txt").readlines()]))
w2idx = {w: idx for idx, w in enumerate(instruction_words)}
dim = max(w2idx.values()) + 1
eval_results = json.load(open("data/results.json"))
X = list()
y = list()
for row in eval_results:
    x = np.zeros(dim)
    x[[w2idx[w.lower()] for w in row["instruction"].split(" ")]] = 1.
    X.append(x)
    y.append([float(row["score"])])

X = np.stack(X)
y = np.concatenate(y)
print(X.shape)
reg = LinearRegression().fit(X, y)
print(reg.score(X, y))

clf = Ridge(alpha=5.0)
clf.fit(X, y)
print(clf.score(X, y))
print(clf.coef_.shape)
topk = 100
for coef, w in list(sorted(zip(clf.coef_, instruction_words), key=lambda x: -x[0]))[:topk]:
    print(w, end=" ")
print()
print("=" * 30)
for coef, w in list(sorted(zip(clf.coef_, instruction_words), key=lambda x: x[0]))[:topk]:
    print(w, end=" ")