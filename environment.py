class Environment:
    def __init__(self):
        self.position = 0
        self.goal = 4

    def reset(self):
        self.position = 0
        return self.position

    def step(self, action):
        if action == 1:
            self.position += 1
        elif action == 0:
            self.position -= 1

        if self.position == self.goal:
            reward = 10
            done = True
        else:
            reward = -1
            done = False

        return self.position, reward, done