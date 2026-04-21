
class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, name, func):
        self.tools[name] = func

    def execute(self, name, *args, **kwargs):
        if name in self.tools:
            return self.tools[name](*args, **kwargs)
        raise ValueError(f'Tool {name} not found')
