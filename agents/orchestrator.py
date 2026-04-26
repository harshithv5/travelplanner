from strands import Agent, tool
from strands.models.ollama import OllamaModel

from guardrails import Guardrails
from memory.short_term import ShortTermMemory

_memory = ShortTermMemory()
_guardrails = Guardrails()


@tool
def discover_destination(query: str) -> str:
    """
    Research a destination: geocode it to confirm it exists and retrieve route info
    between locations. Call this for any destination or routing question.

    Args:
        query: Natural language question about a destination or route,
               e.g. "Where is Coorg?" or "Route from Bangalore to Goa".

    Returns:
        Coordinates, distances, and travel times as a formatted string.
    """
    from agents.discovery import discovery_agent
    return str(discovery_agent(query))


@tool
def plan_budget(query: str) -> str:
    """
    Search hotels and estimate accommodation costs for a city stay.
    Call this once you know the destination city and travel dates.

    Args:
        query: Natural language budget request, e.g.
               "Find hotels in Goa from 2026-06-01 to 2026-06-04 for 2 adults".

    Returns:
        A formatted hotel comparison table with prices in INR.
    """
    from agents.budget import budget_agent
    return str(budget_agent(query))


@tool
def critique_and_refine(plan: str) -> str:
    """
    Critique a draft travel itinerary and return a polished, improved version.
    Always call this as the final step before presenting the plan to the user.

    Args:
        plan: The draft travel plan text to evaluate and rewrite.

    Returns:
        A critique section followed by a fully refined day-by-day itinerary.
    """
    from agents.critic_writer import critic_writer_agent
    return str(critic_writer_agent(f"Critique and rewrite the following travel plan:\n\n{plan}"))


SYSTEM_PROMPT = """You are TravelStack — an intelligent travel planning assistant that orchestrates specialist agents to build complete, personalised trip itineraries.

## Your specialist agents
- **discover_destination**: Verifies locations and calculates driving routes.
- **plan_budget**: Finds hotels and estimates accommodation costs.
- **critique_and_refine**: Critiques a draft plan and rewrites it into a polished itinerary.

## Your planning workflow
1. **Understand** — Extract destination(s), travel dates, number of travellers, and budget. If any are missing, ask one targeted clarifying question, then proceed with reasonable assumptions.
2. **Discover** — Call `discover_destination` for each city mentioned to confirm it exists and get route distances.
3. **Budget** — Call `plan_budget` for each destination city with check-in and check-out dates.
4. **Draft** — Synthesise discovery and budget data into a coherent day-by-day itinerary draft.
5. **Refine** — Pass the draft to `critique_and_refine` to produce the final polished plan.
6. **Respond** — Present the refined plan clearly to the user.

## Rules
- Never skip the discovery step — always confirm locations exist before planning.
- Never skip critique_and_refine — always refine the draft before presenting.
- Maintain context: reference what was discussed earlier in the conversation.
- If a tool returns an error, explain the issue and suggest a fix."""

orchestrator_agent = Agent(
    model=OllamaModel(model_id="qwen3:4b"),
    tools=[discover_destination, plan_budget, critique_and_refine],
    system_prompt=SYSTEM_PROMPT,
)


def run(user_input: str) -> str:
    user_input = _guardrails.validate_input(user_input)
    history = _memory.get_context()
    context_str = "\n".join(
        f"User: {h['user']}\nAssistant: {h['response']}" for h in history
    )
    prompt = f"{context_str}\nUser: {user_input}" if context_str else user_input
    response = str(orchestrator_agent(prompt))
    _memory.save(user_input, response)
    return _guardrails.validate_output(response)
