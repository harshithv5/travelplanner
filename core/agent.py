from core.llm import LLMClient
from core.memory import Memory
from core.tools import ToolRegistry
from core.guardrails import Guardrails

class TravelAgent:
    def __init__(self):
        self.llm = LLMClient()
        self.memory = Memory()
        self.tools = ToolRegistry()
        self.guardrails = Guardrails()

    def run(self, user_input: str):
        user_input = self.guardrails.validate_input(user_input)
        context = self.memory.get_context()
        response = self.llm.generate(user_input, context)
        self.memory.save(user_input, response)
        return self.guardrails.validate_output(response)
