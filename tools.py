class Tools:
    def calculate(self, expression):
        try:
            allowed = "0123456789+-*/(). "

            if not all(character in allowed for character in expression):
                return "Invalid expression."

            result = eval(expression)
            return result

        except Exception:
            return "Could not calculate the expression."