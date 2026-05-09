from base import AbstractTokenizer


class NaiveTokenizer(AbstractTokenizer):
    def __init__(self):
        super().__init__()

    def encode(self, text, allowed_special=None):

        tokens = []
        for char in text:
            if char in self.inv_vocab:
                tokens.append(self.inv_vocab[char])

        for i in range(len(self.bpe_merges)):
            new_tokens = []
            for j in range(len(tokens)-1):
                if tuple([tokens[j], tokens[j+1]]) in self.bpe_merges:
                    new_tokens.append(self.bpe_merges[tuple([tokens[j], tokens[j+1]])])
                    j += 1
                else:
                    new_tokens.append(tokens[j])
            tokens = new_tokens
        return tokens

    def train(self, training_data):
        pass
