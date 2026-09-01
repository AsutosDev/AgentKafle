class Learner:
    def classify(self, information):
        text = information.lower().strip()

        question_patterns = [
            "what ",
            "why ",
            "how ",
            "when ",
            "where ",
            "who ",
            "which ",
            "can you ",
            "could you ",
            "do you ",
            "does ",
            "is ",
            "are ",
            "am i "
        ]

        for pattern in question_patterns:
            if text.startswith(pattern):
                return "question"

        learning_patterns = [
            "my ",
            "i am ",
            "i'm ",
            "i like ",
            "i love ",
            "i prefer ",
            "i want ",
            "i need ",
            "remember "
        ]

        for pattern in learning_patterns:
            if pattern in text:
                return "fact"

        return "conversation"

    def should_learn(self, information):
        return self.classify(information) == "fact"