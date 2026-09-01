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
        self.data.append(information)
        self.save()

    def recall(self):
        return self.data