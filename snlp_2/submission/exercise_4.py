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
        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            last_logits = outputs.logits[:, -1, :]  # Take the logits of the last token
        predictions = torch.nn.functional.softmax(last_logits, dim=-1)
        return predictions

    def decode(self, indices):
        return self.tokenizer.decode(indices)

    def greedy_sample(self, text):
        """
        Takes a context and greedily returns the most likely next token
        """

        # your code here
        probabilities = self.forward(text)
        #  return token index of max probability
        next_token = torch.argmax(probabilities)

        return self.decode(next_token)

    def random_sample(self, text):

        # your code here
        probabilities = self.forward(text)
        #  return a single random token index from probability distribution
        next_token = torch.multinomial(probabilities, num_samples=1).item()

        return self.decode(next_token)

    def rejection_sample(self, text):
        """
        Generate the next token using rejection sampling
        """
        probabilities = self.forward(text)
        # Keep trying until we accept a token
        while True:
            # Step 1: Generate a random number
            r = torch.rand(1).item()

            # Step 2: Randomly sample a token based on the output probability distribution
            sampled_indices = torch.multinomial(probabilities, num_samples=1)
            sampled_index = sampled_indices.item()

            # Step 3: Check if the random number is less than the token's probability
            if r < probabilities[0, sampled_index].item():
                next_token = sampled_index
                break

        return self.decode([next_token])

