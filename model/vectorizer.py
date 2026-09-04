import torch


class BagOfWordsVectorizer:
    def __init__(self, dataset):
        self.words = dataset.words
        self.intents = dataset.intents

    def vectorize(self, tokens):
        vector = [0] * len(self.words)

        for token in tokens:
            if token in self.words:
                index = self.words.index(token)
                vector[index] = 1

        return torch.tensor(vector, dtype=torch.float32)

    def encode_intent(self, intent):
        return torch.tensor(
            self.intents.index(intent),
            dtype=torch.long
        )

    def prepare_data(self, dataset):
        X = []
        y = []

        for tokens, intent in dataset.samples:
            X.append(self.vectorize(tokens))
            y.append(self.encode_intent(intent))

        return torch.stack(X), torch.stack(y)