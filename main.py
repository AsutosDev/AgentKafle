from environment import Environment

env = Environment()

state = env.reset()

print("Starting position:", state)

state, reward, done = env.step(1)

print("New position:", state)
print("Reward:", reward)
print("Done:", done)