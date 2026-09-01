import json
import os


class Memory:
    def __init__(self, filename="memory.json"):
        self.filename = filename
        self.data = self.load()

    def load(self):
        if not os.path.exists(self.filename):
            return []

        with open(self.filename, "r") as file:
            return json.load(file)

    def save(self):
        with open(self.filename, "w") as file:
            json.dump(self.data, file, indent=4)

    def remember(self, information):
        normalized_information = self.normalize(information)

        for memory in self.data:
            if self.normalize(memory) == normalized_information:
                return False  # Avoid storing duplicate information

        self.data.append(information)
        self.save()
        return True  # Successfully remembered new information

    def recall(self):
        return self.data

    def normalize(self, text):
        return text.lower().strip(".,!?;:")

    def search(self, query):
        stop_words = {
            "what",
            "is",
            "the",
            "a",
            "an",
            "my",
            "your",
            "i",
            "you",
            "of",
            "to",
            "in",
            "for",
            "and",
        }

        query_words = {
            word for word in self.normalize(query).split() if word not in stop_words
        }

        results = []

        for memory in self.data:
            memory_words = {
                word
                for word in self.normalize(memory).split()
                if word not in stop_words
            }

            score = len(query_words & memory_words)

            if score > 0:
                results.append((score, memory))

        results.sort(reverse=True)

        return [memory for score, memory in results]
