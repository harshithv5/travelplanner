class ShortTermMemory:
    def __init__(self, max_history: int = 5):
        self.history = []
        self.max_history = max_history

    def get_context(self) -> list:
        return self.history[-self.max_history:]

    def save(self, user_input: str, response: str):
        self.history.append({"user": user_input, "response": response})
