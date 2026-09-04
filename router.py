class Router:
    def decide(self, input_text):
        text = input_text.lower().strip()

        calculator_words = [
            "calculate",
            "calculator",
        ]

        memory_words = [
            "remember",
            "my name",
            "my favorite",
            "i like",
        ]

        conversation_words = [
            "hello",
            "hi",
            "hey",
            "good morning",
            "good night",
            "how are you",
        ]

        if any(word in text for word in calculator_words):
            return "calculator"

        if any(char.isdigit() for char in text):
            if any(operator in text for operator in ["+", "-", "*", "/"]):
                return "calculator"

        if any(word in text for word in memory_words):
            return "memory"

        if any(word in text for word in conversation_words):
            return "conversation"

        return "none"
