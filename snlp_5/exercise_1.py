import math
from collections import Counter
from typing import List, Union
from collections import defaultdict

from tokenizers import Tokenizer
from morfessor.baseline import BaselineModel


class TokenizerEntropy:
    def tokenize_bpe(self, tokenizer: Tokenizer, text: str) -> List[str]:
        """
        Takes the BPE tokenizer and a text and returns the list of tokens.

        params:
        - tokenizer: The pre-trained BPE tokenizer
        - text: The input text to tokenize

        returns a list of tokens
        """
        # ====================================
        # Your code here

        # word_freqs = defaultdict(int)
        # merges = {}
        # alphabet = []
        # vocab_size = 200 #arbitary vocab size for merges

        # new_words = text.split()
        # for word in new_words:
        #     word_freqs[word] += 1

        
        # for word in word_freqs.keys():
        #     for letter in word:
        #         if letter not in alphabet:
        #             alphabet.append(letter)
        # alphabet.sort()

        # vocab = ["<|endoftext|>"] + alphabet.copy()

        # # calculating corpus each word slpits
        # splits = {word: [c for c in word] for word in word_freqs.keys()}

        # while len(vocab) < vocab_size:
        #     pair_freqs = self.compute_pair_freqs(splits,word_freqs)
        #     best_pair = ""
        #     max_freq = None
        #     for pair, freq in pair_freqs.items():
        #         if max_freq is None or max_freq < freq:
        #             best_pair = pair
        #             max_freq = freq
        #     splits = self.merge_pair(*best_pair, splits,word_freqs)
        #     merges[best_pair] = best_pair[0] + best_pair[1]
        #     vocab.append(best_pair[0] + best_pair[1])

        # ====================================
        return tokenizer.encode(text).tokens

    def tokenize_morfessor(self, tokenizer: BaselineModel, text: str) -> List[str]:
        """
        Takes the Morfessor tokenizer and a text and returns the list of tokens.

        params:
        - tokenizer: The pre-trained Morfessor tokenizer
        - text: The input text to tokenize

        returns a list of tokens
        """
        # ====================================
        # Your code here
        tokens = []
        for word in text.split():
            tokens.extend(tokenizer.viterbi_segment(word)[0])
        return tokens
        # ====================================
        raise NotImplementedError

    def get_probs(self, tokens: List[str]):
        """
        Takes a list of tokens and compute the probability distribution of the tokens.

        params:
        - tokens: A list of tokens

        returns a dictionary of token probabilities i.e. {token: probability, ...}
        """
        # ====================================
        # Your code here

        token_freq = Counter(tokens)
        total_tokens = len(tokens)

        token_probs = {token: count / total_tokens for token, count in token_freq.items()}

        # ====================================
        return token_probs

    def compute_entropy(
        self, text: str, tokenizer: Union[Tokenizer, BaselineModel]
    ) -> float:
        """
        Takes the input text and the tokenizer and returns the entropy of the text.

        params:
        - text: The input text
        - tokenizer: The pre-trained tokenizer (BPE or Morfessor)

        returns the entropy of the text
        """
        # tokenize the input text
        if isinstance(tokenizer, Tokenizer):
            tokens = self.tokenize_bpe(tokenizer, text)
        elif isinstance(tokenizer, BaselineModel):
            tokens = self.tokenize_morfessor(tokenizer, text)
        else:
            raise ValueError("Tokenizer not supported.")

        # ====================================
        # Your code here

        # get the probabilities of each token
        prob_dist = self.get_probs(tokens)

        # Compute the entropy
        entropy = -sum(prob * math.log2(prob) for prob in prob_dist.values() if prob != 0)

        # ====================================
        return entropy
    
    #  To compute pair frequency for BPE implementation
    def compute_pair_freqs(splits,word_freqs):
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    #  To merge the most occured pair in splits
    def merge_pair(a, b, splits,word_freqs):
        for word in word_freqs:
            split = splits[word]
            if len(split) == 1:
                continue

            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2 :]
                else:
                    i += 1
            splits[word] = split
        return splits
