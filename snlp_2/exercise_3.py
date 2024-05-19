import random


class MillersModel:
    """Implements the Miller's Model"""

    _char_to_prob = {
        " ": 0.2,
        "e": 0.08929,
        "a": 0.06798,
        "r": 0.06065,
        "i": 0.06036,
        "o": 0.05731,
        "t": 0.05561,
        "n": 0.05323,
        "s": 0.04588,
        "l": 0.04391,
        "c": 0.03631,
        "u": 0.02905,
        "d": 0.02707,
        "p": 0.02534,
        "m": 0.0241,
        "h": 0.02402,
        "g": 0.01976,
        "b": 0.01658,
        "f": 0.0145,
        "y": 0.01422,
        "w": 0.01032,
        "k": 0.00882,
        "v": 0.00806,
        "x": 0.00232,
        "z": 0.00218,
        "j": 0.00158,
        "q": 0.00157,
    }

    def sample(self):
        """Sample a character based on the probability"""
        return random.choices(
            list(self._char_to_prob.keys()), weights=list(self._char_to_prob.values())
        )[0]

    def get_prob(self, char):
        """
        Get the probability of a character

        params
            char: The character to get the probability of
        """
        return self._char_to_prob.get(str.lower(char), 0)

    def compute_perplexity(self, text):
        """
        Compute the perplexity of a text

        params
            text: The text to compute the perplexity of
        """
        
        # your code here

        # Perplexity(w)= (i=1∏N 1/​P(wi​)) ** 1/​N
        
        characters = list(text)

        pwi = 1

        for char in characters:
            pwi *= self.get_prob(char)

        perplexity_w = pow((1 / pwi), -(1 / len(characters)))

        return perplexity_w
