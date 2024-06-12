import math
from tqdm import tqdm
from collections import defaultdict

from typing import List, Union


class LanguageModel:
    def __init__(self, n_gram: int = 1) -> None:
        self.n_gram = n_gram # Keep track of the n-gram value
        self.probs = {} # To store the probabilities of the n-grams i.e. {tuple(w1, w2, .., wn): probability, ...}
        self.n_gram_counts = {} # To store the counts of the n-grams i.e. {tuple(w1, w2, .., wn): count, ...}
        self.n_minus_1_gram_counts = {} # To store the counts of the n-1 grams i.e. {tuple(w1, w2, .., wn-1): count, ...}
        self.unigram_counts = {} # To store the counts of the unigrams i.e. {tuple(w,): count, ...}

    def get_counts(self, tokens: List[str], n: int = 1) -> dict:
        """
        Takes a list of tokens and returns the n-gram counts.

        params:
        - tokens: A list of tokens
        - n: The value of n for n-grams

        returns a dictionary of n-gram counts i.e. {(w1, w2, .., wn): count, ...}

        Example:
        tokens = ['this', 'is', 'a', 'sentence']
        get_counts(tokens, n=2)
        Output: {('this', 'is'): 1, ('is', 'a'): 1, ('a', 'sentence'): 1}
        """
        n_gram_counts = defaultdict(int)

        # ====================================
        # Your code here
        # Iterate over the tokens to generate n-grams and count their occurrences
        for i in range(len(tokens) - n + 1):

            n_gram = tuple(tokens[i:i + n])
            n_gram_counts[n_gram] += 1

        # ====================================
        return n_gram_counts

    def train(self, tokens: List[str]) -> None:
        """
        Takes a text and trains the language model.
        Training here means computing the probabilities of the n-grams and storing them in self.probs.

        params:
        - tokens: A list of tokens

        returns None
        """

        self.n_gram_counts = self.get_counts(tokens, n=self.n_gram)  # N(h, w)
        self.n_minus_1_gram_counts = self.get_counts(tokens, n=self.n_gram - 1)  # N(h)
        self.unigram_counts = self.get_counts(tokens, n=1)  # N(w)

        # ====================================
        # Your code here

        # Calculate probabilities
        for n_gram, count in self.n_gram_counts.items():
            if self.n_gram == 1:
                # then n-gram is same is unigram so count(w)/ N - total words
                # For unigram probabilities
                self.probs[n_gram] = count / sum(self.unigram_counts.values())
            else:
                #  split the n gram and look for probbaility in previous gram (is,the,dog)
                n_minus_1_gram = n_gram[:-1] # (is,the) from n_minus_1_gram
                # count (is,the,dog) / count (is,the)
                self.probs[n_gram] = count / self.n_minus_1_gram_counts[n_minus_1_gram]

    def generate(self, history_tokens: List[str]) -> str:
        """
        Takes a list of tokens and returns the most likely next token.
        Return None if the history is not present in the model.

        params:
        - history_tokens: A list of tokens

        returns the next token
        """

        # Convert it into a tuple, in case it's already not
        history = tuple(history_tokens)

        if len(history) != self.n_gram - 1:
            # If history is longer than what's required for our n-gram model
            # simply take the last n-1 tokens
            history = history[-(self.n_gram - 1):]

        max_prob = 0
        next_token = None

        # Search through n_grams to find the best next token
        for n_gram in self.probs:
            if n_gram[:-1] == history:
                if self.probs[n_gram] > max_prob:
                    max_prob = self.probs[n_gram]
                    next_token = n_gram[-1]

        return next_token

    def get_smoothed_probs(self, n_gram: List[str], d: float = 0.1) -> float:
        """
        Takes a n-gram and returns the smoothed probability using absolute discounting.

        params:
        - n_gram: A list/tuple of tokens (w1, w2, .., wn)
        - d: The discounting factor

        returns the smoothed probability
        """

        n_gram = tuple(n_gram)

        history = n_gram[:-1]
        w = n_gram[-1]

        # ====================================
        # Your code here
        # Compute step by step to prevent errors

        # Computing the lambda(.) value
        total_history_tokens = sum(self.n_minus_1_gram_counts.values())
        total_tokens = sum(self.unigram_counts.values())
        # unique_history_tokens = len(self.n_minus_1_gram_counts)
        # unique_tokens = len(self.unigram_counts)
        # lambda_ = d / total_tokens * unique_tokens

        # Computing the lambda(w_{i-1}) value
        # lambda_h = d / total_history_tokens * unique_history_tokens if total_history_tokens > 0 else 0
        N1_plus = sum(1 for h in self.n_minus_1_gram_counts if tuple(h[:-1]) == history)
        lambda_h = d * N1_plus / total_history_tokens if total_history_tokens > 0 else 0

        N1 = len(self.unigram_counts)
        lambda_ = d * N1 / total_tokens if total_tokens > 0 else 0

        # Computing P_abs(w_i)
        # P_abs_w_i = max(self.unigram_counts.get((w,), 0) - d, 0) / total_tokens + lambda_ * (1 / unique_tokens)
        P_abs_w_i = max(self.unigram_counts.get((w,), 0) - d, 0) / total_tokens if total_tokens > 0 else 0
        P_abs_w_i += lambda_ * (1 / N1)

        # Computing P_abs(w_i | w_{i-1})
        P_abs_w_i_given_h = max(self.n_gram_counts.get(n_gram, 0) - d, 0) / self.n_minus_1_gram_counts.get(history, 0) if self.n_minus_1_gram_counts.get(history, 0) > 0 else 0
        P_abs_w_i_given_h += lambda_h * P_abs_w_i
        
        # ====================================

        return P_abs_w_i_given_h

    def generate_absolute_smoothing(self, history_tokens: List[str], d: float = 0.1) -> str:
        """
        Takes a list of tokens and returns the most likely next token using absolute discounting.

        params:
        - history_tokens: A list of tokens (w1, w2, .., wn-1) for which we want to predict the next token
        - d: The discounting factor

        returns the next token
        """

        # Convert it into a tuple, in case it's already not
        history_tokens = tuple(history_tokens)

        if len(history_tokens) != self.n_gram - 1:
            # If history is longer than what's required for our n-gram model
            # simply take the last n-1 tokens
            history_tokens = history_tokens[-(self.n_gram - 1) :]

        max_prob = 0
        next_token = None

        # ====================================
        # Your code here
        for token in self.unigram_counts:
            potential_ngram = history_tokens + (token,)
            prob = self.get_smoothed_probs(potential_ngram, d)
            if prob > max_prob:
                max_prob = prob
                next_token = token
        # ====================================

        return next_token

    def perplexity(self, tokens: List[str], n_words: int, d: float = None) -> float:
        """
        Takes a list of tokens and returns the perplexity of the language model (per word)
        Remember to normalize the probabilities by the number of words (not number of tokens or n-grams, see formula) in the text.

        params:
        - tokens: A list of tokens
        - n_words: The number of words in the text
        - d: The discounting factor for absolute discounting (only for absolute discounting, otherwise None)

        returns the perplexity of the language model
        """

        log_prob = 0
        ngrams = self.get_counts(tokens, n=self.n_gram).keys()

        total_unigram_count = sum(self.unigram_counts.values())

        for ngram in ngrams:
            if ngram in self.probs:
                prob = self.probs[ngram]
            else:
                if d is not None and total_unigram_count > 0:
                    prob = max(d / total_unigram_count, 1e-10)  # Avoid zero by using a small constant
                else:
                    
                    prob = 1e-10  # Fallback to avoid log(0)

            log_prob += math.log(prob)

        log_prob_avg = log_prob / n_words

        perplexity = math.exp(-log_prob_avg)

        return perplexity
        # ====================================