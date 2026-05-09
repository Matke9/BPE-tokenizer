from abc import abstractmethod, ABC

import json


class AbstractTokenizer(ABC):
    def __init__(self):
        self.vocab = {}
        self.inv_vocab = {}
        self.bpe_merges = {}
        self.bpe_ranks = {}

    @abstractmethod
    def encode(self, text, allowed_special):
        pass

    @abstractmethod
    def train(self, training_data):
        pass

    def decode(self, tokens):
        text = ''
        for token in tokens:
            if token in self.vocab:
                text += self.vocab[token]
            else:
                text += token
        text = text.replace('Ġ', ' ')
        return text

    def save_vocab_and_merges(self, vocab_path, merges_path):
        with open(vocab_path, "w", encoding="utf-8") as file:
            json.dump(self.vocab, file, ensure_ascii=False, indent=2)
        with open(merges_path, "w", encoding="utf-8") as file:
            merges_list = [{"pair": list(pair), "new_id": new_id}
                           for pair, new_id in self.bpe_merges.items()]
            json.dump(merges_list, file, ensure_ascii=False, indent=2)

    def load_vocab_and_merges(self, vocab_path, merges_path):
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            self.vocab = {int(k): v for k, v in loaded_vocab.items() }
            self.inv_vocab = {v: int(k) for k, v in loaded_vocab.items() }
        with open(merges_path, "r", encoding="utf-8") as file:
            loaded_merges = json.load(file)
            for merge in loaded_merges:
                pair = tuple(merge["pair"])
                self.bpe_merges[pair] = merge["new_id"]
