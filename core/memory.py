class Memory:
    def __init__(self):
        self.history = []

    def get_context(self):
        return self.history[-5:]

    def save(self, user_input, response):
        self.history.append({'user': user_input, 'response': response})
