import json
import re


class IntentDataset:
    def __init__(self, filename="data/intents.json"):
        self.filename = filename
        self.data = self.load()

        self.words = []
        self.intents = []
        self.samples = []

        self.prepare()

    def load(self):
        with open(self.filename, "r") as file:
            return json.load(file)

    def tokenize(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def prepare(self):
        for intent, examples in self.data.items():
            if intent not in self.intents:
                self.intents.append(intent)

            for example in examples:
                tokens = self.tokenize(example)

                self.samples.append((tokens, intent))

                for word in tokens:
                    if word not in self.words:
                        self.words.append(word)

        self.words.sort()
        self.intents.sort()

    def summary(self):
        return {
            "intents": self.intents,
            "vocabulary_size": len(self.words),
            "samples": len(self.samples),
        }