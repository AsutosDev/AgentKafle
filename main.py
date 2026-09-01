from agent import Agent


def main():
    agent = Agent()

    print(f"{agent.name} is online.")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("AgentKafle shutting down.")
            break

        if user_input.lower().startswith("learn "):
            information = user_input[6:]
            response = agent.learn(information)
        elif user_input.lower() == "memory":
            response = agent.recall()
        else:
            response = agent.think(user_input)

        print(f"AgentKafle: {response}")


if __name__ == "__main__":
    main()