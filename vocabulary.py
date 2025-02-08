from data.vocabularies.flan import PATTERNS
import re
from nltk.tokenize import word_tokenize
import nltk
# nltk.download('punkt_tab')


def get_flan_vocab():
    vocab = set()
    for row in PATTERNS.values():
        for col in row:
            for s in col:
                s_prepped = re.sub(r'[()_:"{},.;@#?!&$]+', ' ', s)
                words = word_tokenize(s_prepped)
                for w in words:
                    w = w.strip()
                    if w.isalpha() and len(w) > 1:
                        vocab.add(w.lower())
    return list(vocab)
