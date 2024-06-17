from collections import defaultdict
import math

class CountTree():
    def __init__(self, n=4):
        self.depth = n
        if n == 1:
            self.nodes = defaultdict(int)
        else:
            self.nodes = defaultdict(lambda: CountTree(n=n-1))

    def add(self, ngram):
        reversed_ngram = tuple(reversed(ngram))
        self._add_helper(reversed_ngram)

    def _add_helper(self, ngram):
        if self.depth == 1:
            self.nodes[ngram] += 1
        else:
            first, *rest = ngram
            if rest:
                self.nodes[first]._add_helper(tuple(rest))
            else:
                self.nodes[first]._add_helper(first)

    def get(self, ngram):
        reversed_ngram = tuple(reversed(ngram))
        return self._get_helper(reversed_ngram)

    def _get_helper(self, ngram):
        if not ngram:
            return sum(self.nodes.values()) if self.depth == 1 else sum(node._get_helper("") for node in self.nodes.values())
        else:
            if self.depth == 1:
                return self.nodes.get(ngram, 0)
            else:
                first, *rest = ngram
                if first in self.nodes:
                    if rest:
                        return self.nodes[first]._get_helper(tuple(rest))
                    else:
                        return self.nodes[first]._get_helper("")
                else:
                    return 0

    def cond_prob(self, word, history):
        full_ngram = tuple(reversed(history + (word,)))
        history = tuple(reversed(history))
        count_word_given_history = self.get(full_ngram)
        count_history = self.get(history)
        if count_history > 0:
            return count_word_given_history / count_history
        else:
            return 0.0

    def perplexity(self, ngrams, vocab):
        log_sum = 0
        for ngram in ngrams:
            p4 = self.cond_prob(ngram[-1], ngram[:-1])
            p0 = 1 / vocab
            probability = 0.75 * p4 + 0.25 * p0
            log_sum += math.log2(probability)
        return 2 ** (-log_sum / len(ngrams))

    def prune(self, k):
        if self.depth == 1:
            for word, count in list(self.nodes.items()):
                if count <= k:
                    del self.nodes[word]
        else:
            for word, subtree in list(self.nodes.items()):
                if isinstance(subtree, CountTree):
                    subtree.prune(k)
                    if (hasattr(subtree, '_get_helper')):
                        total = subtree._get_helper("")
                    else:
                        total = subtree
                    if total <= k:
                        self.nodes[word] = total
                else:
                    if subtree <= k:
                        del self.nodes[word]

