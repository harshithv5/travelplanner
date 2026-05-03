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
# Summarizer — lightweight local model to compress Tavily results before
# passing them to the main LLM, keeping the context window lean.
# ---------------------------------------------------------------------------
_summarizer_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

def summarize_tool_output(city: str, category: str, results: list) -> dict:
    """
    Compress raw Tavily results with qwen3:0.6b.

    Returns:
        {
          "summary": compact text for the main LLM context,
          "places":  [{place, description, url}]  — structured extracted entries
        }
    """
    if not results:
        return {"summary": f"No results for '{category}' in {city}.", "places": []}

    raw = "\n---\n".join(
        f"Title: {r['title']}\nSnippet: {r['content'][:400]}\nURL: {r['url']}"
        for r in results
    )

    prompt = (
        f"Extract specific named places from these travel articles about '{category}' in {city}.\n"
        f"For each place output exactly: PLACE | one-sentence reason to visit | source URL\n"
        f"Rules: real place names only (not article titles), max 5 places, one per line.\n\n"
        f"{raw}\n\nFormat: PLACE | DESCRIPTION | URL"
    )

    try:
        resp = _summarizer_client.chat.completions.create(
            model="qwen3:0.6b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.1
        )
        text = resp.choices[0].message.content.strip()
    except Exception:
        text = ""

    places = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3 and parts[0] and not parts[0].startswith(("#", "-", "*")):
            url = parts[2] if parts[2].startswith("http") else results[0]["url"]
            places.append({"place": parts[0], "description": parts[1], "url": url})

    # Fallback: parsing failed — use raw titles so we never lose data
    if not places:
        places = [
            {"place": r["title"], "description": r["content"][:150], "url": r["url"]}
            for r in results
        ]

    summary = (
        f"[{category} | {city}] {len(places)} places found: "
        + ", ".join(p["place"] for p in places)
    )
    return {"summary": summary, "places": places}
