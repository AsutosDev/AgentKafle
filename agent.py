from environment import Environment
from memory import Memory


class Agent:
    def __init__(self, name="AgentKafle"):
        self.name = name
        self.memory = Memory()
        self.environment = Environment()

    def observe(self):
        return self.environment.observe()

    def think(self, input_text):
        return f"{self.name} is thinking about: {input_text}"

    def learn(self, information):
        self.memory.remember(information)
        return f"{self.name} learned: {information}"

    def recall(self):
        return self.memory.recall()