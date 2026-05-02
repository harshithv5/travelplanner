from langfuse import Langfuse

langfuse = Langfuse()

def get_tracer():
    return langfuse