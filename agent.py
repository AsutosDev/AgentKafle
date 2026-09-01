import re

from router import Router
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
        self.router = Router()

    def observe(self):
        return self.environment.observe()

    def think(self, input_text):
        classification = self.learner.classify(input_text)

        if classification == "fact":
            if self.learner.should_learn(input_text):
                learned = self.learn(input_text)
                if learned:
                    return f"{self.name} learned: {input_text}"
                return f"{self.name} already knows: {input_text}"

        if classification == "question":
            memories = self.memory.search(input_text)

            if memories:
                return f"I remember: {memories[0]}"

            if "my name" in input_text.lower():
                return f"Your name is {self.name}."

        tool = self.router.decide(input_text)

        if tool == "calculator":
            expression = re.sub(r"[^0-9+\-*/(). ]", "", input_text)

            try:
                result = self.tools.calculate(expression)

                if result != "Invalid expression.":
                    return f"I calculated: {result}"

                return "Invalid expression."

            except Exception as e:
                return f"Error calculating the expression: {str(e)}"

        return f"{self.name} is thinking about: {input_text}"

    def learn(self, information):
        return self.memory.remember(information)

    def search_memory(self, query):
        return self.memory.search(query)
