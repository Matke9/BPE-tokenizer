from abc import abstractmethod, ABC


class AbstractTokenizer(ABC):
    def __init__(self):
        self.vocab = {}
        self.inv_vocab = {}
        self.bpe_merges = {}
        self.bpe_ranks = {}

    @abstractmethod
    def encode(self, text):
        pass

    @abstractmethod
    def train(self, training_data):
        pass

    @abstractmethod
    def merge(self, token1, token2):
        pass

    def decode(self, tokens):
        raise NotImplementedError("Decode method not implemented.") #implement in next session

