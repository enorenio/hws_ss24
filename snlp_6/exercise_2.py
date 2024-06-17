from collections import Counter
from math import ceil, log2, prod, floor
from nltk.tokenize import word_tokenize
import string

def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    return tokens


class SmoothingCounter:
    def __init__(self, text,alpha=0):
        """ 
        :param text: preprocessed corpus
        :param d: discounting parameter, this is fixed
        :param alpha :  Smoothing factor for laplace smoothing
        :function kncounts(bigram) : Calculates the log probability of a bigram based on Kneser-Ney Counts
        :function logprob_alpha(bigram) : Calculates the log probabillty of a bigram based on Laplace smoothing
        """
        self.alpha = alpha
        self.d = 0.75
        self.text = text
        self.unigram_counts = Counter(text)
        self.bigram_counts = Counter(zip(text, text[1:]))
        self.trigram_counts = Counter(zip(text, text[1:], text[2:]))
        self.vocab_size = len(set(text))
        self.total_bigrams = sum(self.bigram_counts.values())
        self.total_trigrams = sum(self.trigram_counts.values())

    def knprob_bigram(self,bigram):
        '''returns the log probability of a bigram with counts adjusted for Knser-Ney Smoothing'''  
        w2, w3 = bigram
        N_plus_cont = sum(1 for u in set(self.text) if (u, w3) in self.bigram_counts)
        lambda_w2 = (self.d / self.unigram_counts[w2]) * N_plus_cont
        P_cont = max(self.bigram_counts[bigram] - self.d, 0) / self.unigram_counts[w2]
        P_backoff = N_plus_cont / self.vocab_size
        return P_cont + lambda_w2 * P_backoff

    def knprob_trigram(self,trigram):
        '''returns the log probability of a trigram with counts adjusted for Knser-Ney Smoothing'''  
        w1, w2, w3 = trigram
        N_plus_cont = sum(1 for u in set(self.text) if (u, w2, w3) in self.trigram_counts)
        lambda_w1w2 = (self.d / self.bigram_counts[(w1, w2)]) * N_plus_cont
        P_cont = max(self.trigram_counts[trigram] - self.d, 0) / self.bigram_counts[(w1, w2)]
        P_backoff = self.knprob_bigram((w2, w3))
        return P_cont + lambda_w1w2 * P_backoff


    def prob_alpha_bigram(self,bigram):
        '''returns the log probability of a bigram with counts adjusted for add-alpha Smoothing'''  
        w2, w3 = bigram
        return (self.bigram_counts[bigram] + self.alpha) / (self.unigram_counts[w2] + self.alpha * self.vocab_size)

    def prob_alpha_trigram(self, trigram):
        '''returns the probability of a trigram with counts adjusted for add-alpha Smoothing'''
        w1, w2, w3 = trigram
        return (self.trigram_counts[trigram] + self.alpha) / (self.bigram_counts[(w1, w2)] + self.alpha * self.vocab_size)