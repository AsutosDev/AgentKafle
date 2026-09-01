import re

from environment import Environment
from memory import Memory
from learner import Learner
from tools import Tools


class Agent:
    def __init__(self, name="AgentKafle"):
        self.name = name
        self.memory = Memory()
        self.environment = Environment()
        self.learner = Learner()
        self.tools = Tools()

    def observe(self):
        return self.environment.observe()

    def think(self, input_text):
        classification = self.learner.classify(input_text)

        if classification == "fact":
            if self.learner.should_learn(input_text):
                self.memory.remember(input_text)
                return f"I learned: {input_text}"

        if classification == "question":
            memories = self.memory.search(input_text)

            if memories:
                return f"I remember: {memories[0]}"

            if any(char.isdigit() for char in input_text):
                expression = re.sub(r"[^0-9+\-*/().]", "", input_text)

                result = self.tools.calculate(expression)

                if result != "Invalid expression.":
                    return f"I calculated: {result}"

            return "I don't know the answer to that yet."

        return f"{self.name} is thinking about: {input_text}"

    def learn(self, information):
        self.memory.remember(information)
        return f"{self.name} learned: {information}"

    def recall(self):
        return self.memory.recall()

    def search_memory(self, query):
        return self.memory.search(query)