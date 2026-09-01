class Environment:
    def __init__(self):
        self.state = {}

    def observe(self):
        return self.state

    def update(self, key, value):
        self.state[key] = value