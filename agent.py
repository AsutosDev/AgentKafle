from environment import Environment
from memory import Memory
from learner import Learner


class Agent:
    def __init__(self, name="AgentKafle"):
        self.name = name
        self.memory = Memory()
        self.environment = Environment()
        self.learner = Learner()

    def observe(self):
        return self.environment.observe()

    def think(self, input_text):
        if self.learner.should_learn(input_text):
            self.memory.remember(input_text)
            return f"I learned: {input_text}"

        return f"{self.name} is thinking about: {input_text}"

    def learn(self, information):
        self.memory.remember(information)
        return f"{self.name} learned: {information}"

    def recall(self):
        return self.memory.recall()