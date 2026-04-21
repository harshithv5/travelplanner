class Guardrails:
    def validate_input(self, text: str):
        return text.strip()

    def validate_output(self, text: str):
        return text
