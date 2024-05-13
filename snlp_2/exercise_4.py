import torch
from transformers import GPT2Tokenizer, AutoModelForCausalLM


class GPT2Model:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.model.eval() # eval mode since we don't need to train

    def forward(self, text):
        """
        Take a text and return the probabilities of the next token
        """

        # your code here

        raise NotImplementedError

    def decode(self, indices):
        return self.tokenizer.decode(indices)

    def greedy_sample(self, text):
        """
        Takes a context and greedily returns the most likely next token
        """

        # your code here
        next_token = None

        return self.decode(next_token)

    def random_sample(self, text):

        # your code here
        next_token = None

        return self.decode(next_token)

    def rejection_sample(self, text):

        # your code here
        next_token = None

        return self.decode(next_token)
