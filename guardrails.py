class Guardrails:
    def validate_input(self, text: str) -> str:
        if not text or not text.strip():
            raise ValueError("Input cannot be empty")
        return text.strip()

    def validate_output(self, text: str) -> str:
        return text
