class BaseAgent:
    def run(self, input_data, context=None):
        raise NotImplementedError('Each agent must implement run()')
