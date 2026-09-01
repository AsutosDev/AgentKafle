from environment import Environment


class Agent:
    def __init__(self, name="AgentKafle"):
        self.name = name
        self.memory = []
        self.environment = Environment()

    def remember(self, information):
        self.memory.append(information)

    def observe(self):
        return self.environment.observe()

    def think(self, input_text):
        return f"{self.name} is thinking about: {input_text}"

    def learn(self, information):
        self.remember(information)
        return f"{self.name} learned: {information}"