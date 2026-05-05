import os
from dotenv import load_dotenv
from openai import OpenAI
from strands.models.openai import OpenAIModel
from strands.models.ollama import OllamaModel
from strands.models.litellm import LiteLLMModel
from strands.models.gemini import GeminiModel
from strands.models.mistral import MistralModel

load_dotenv()

# --- Ollama (local) ---
ollama = OllamaModel(
    host="http://localhost:11434",
    model_id="qwen3:4b"
)

# --- Groq via OpenAI-compatible client ---
_groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)
groq = OpenAIModel(
    client=_groq_client,
    model_id="llama-3.3-70b-versatile",
    params={
        "temperature": 0.3,
        "max_tokens": 1024
    }
)

# --- Groq via LiteLLM ---
groq_litellm = LiteLLMModel(
    model_id="groq/meta-llama/llama-4-scout-17b-16e-instruct",
    params={"api_key": os.getenv("GROQ_API_KEY")}
)

# --- Gemini ---
gemini = GeminiModel(
    client_args={"api_key": os.getenv("GEMINI_API_KEY")},
    model_id="gemini-2.0-flash"
)

# --- Mistral ---
mistral = MistralModel(
    client_args={"api_key": os.getenv("MISTRAL_API_KEY")},
    model_id="mistral-small-latest"
)

# --- Cerebras ---
cerebras = OpenAIModel(
    client_args={
        "api_key": os.getenv("CEREBRAS_API_KEY"),
        "base_url": "https://api.cerebras.ai/v1"
    },
    model_id="zai-glm-4.7",
    params={
        "temperature": 0.3
    }
)

# --- Mistral (JSON mode) ---
mistral_json = MistralModel(
    client_args={
        "api_key": os.getenv("MISTRAL_API_KEY"),
        "response_format": {"type": "json_object"}
    },
    model_id="mistral-small-latest"
)

# ---------------------------------------------------------------------------
# Summarizer — lightweight local model to compress tool outputs before
# passing them to the main LLM, keeping the context window lean.
# Uses raw httpx so calls are never picked up by Langfuse instrumentation.
# ---------------------------------------------------------------------------
_SUMMARIZER_URL = "http://localhost:11434/v1/chat/completions"

_DEFAULT_GENERATION_CONFIG = {
    "model":           "qwen3:0.6b",
    "max_tokens":      400,
    "temperature":     0.1,
    "response_format": {"type": "text"},
    "num_ctx":         4096,
}

def summarize_tool_output(prompt: str, generation_config: dict = None) -> str | dict:
    """
    Run a prompt through the local summarizer model.

    Args:
        prompt:            The prompt to send to the model.
        generation_config: Optional overrides — model, max_tokens, temperature,
                           response_format ({"type": "text"} or {"type": "json_object"}).

    Returns:
        Parsed dict if response_format is json_object, otherwise raw text string.
        Returns "" / {} on failure.
    """
    import json
    import httpx

    config = {**_DEFAULT_GENERATION_CONFIG, **(generation_config or {})}
    is_json = config["response_format"].get("type") == "json_object"

    payload = {
        "model":       config["model"],
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  config["max_tokens"],
        "temperature": config["temperature"],
        "options":     {"num_ctx": config["num_ctx"]},
        "stream":      False,
    }
    if is_json:
        payload["response_format"] = {"type": "json_object"}

    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                _SUMMARIZER_URL,
                json=payload,
                headers={"Authorization": "Bearer ollama"},
            )
        text = resp.json()["choices"][0]["message"]["content"].strip()
        if is_json:
            return json.loads(text)
        return text
    except Exception:
        return {} if is_json else ""
