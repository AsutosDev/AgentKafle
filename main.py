from agent import Agent


def main():
    agent = Agent()

    print(f"{agent.name} is online.")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("AgentKafle shutting down.")
            break

        response = agent.think(user_input)
        print(response)


if __name__ == "__main__":
    main()