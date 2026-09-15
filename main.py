"""AgentKafle command-line entry point.

Run with:
    python main.py

The REPL (read-evaluate-print loop) works in two stages:

1. Case commands (create/list/select/view/close) are handled directly by
   AgentKafle and CaseManager — the LLM is not involved, so case data is
   never changed by the model.
2. Anything else is sent to the LLM for reasoning, with the active case
   attached as context.

Type 'help' to see the case commands, or 'exit' to quit.
"""

from agent import AgentKafle


HELP_TEXT = """\
Commands:
  new case <title> [| <description>]   Create a case (becomes active)
  list cases                           List all cases
  open case CASE-001                   Select the active case
  current case                         Show the active case
  close case [CASE-001]                Mark the active case CLOSED
  solve case [CASE-001]                Mark the active case SOLVED
  delete case CASE-001                 Delete a case
  help                                 Show this help
  exit / quit                          Shut down

Anything else is sent to AgentKafle for detective reasoning.\
"""


def main():
    agent = AgentKafle()

    print(f"{agent.name} is online. Type 'help' for commands, 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print(f"{agent.name} shutting down.")
            break

        if user_input.lower() == "help":
            print(HELP_TEXT + "\n")
            continue

        handled, response = agent.handle_command(user_input)
        if not handled:
            response = agent.respond(user_input)

        print(f"{agent.name}: {response}\n")


if __name__ == "__main__":
    main()