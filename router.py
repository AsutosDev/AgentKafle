class Router:
    def decide(self, input_text):
        text = input_text.lower()

        if any(char.isdigit() for char in text):
            return "calculator"

        return None