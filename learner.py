class Learner:
    def should_learn(self, information):
        information = information.lower()

        keywords = [
            "my",
            "i am",
            "i'm",
            "i like",
            "i love",
            "i prefer",
            "i want",
            "i need",
            "remember"
        ]

        return any(keyword in information for keyword in keywords)