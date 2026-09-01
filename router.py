class Router:
    def decide(self, input_text):
        text = input_text.lower().strip()

        if any(char.isdigit() for char in text):
            if any(operator in text for operator in ["+", "-", "*", "/"]):
                return "calculator"

        calculator_words = [
            "calculate",
            "calculator",
        ]

        if any(word in text for word in calculator_words):
            return "calculator"

        return "none"
