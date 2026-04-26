TravelStack — AI Travel Planning Agent

A multi-agent AI system that plans personalised trips with hidden gems, real routes, weather checks, and hotel recommendations. Built with Strands Agents, Gemini Flash 2.0, Qdrant, and free APIs.


What it does
You tell TravelStack where you want to go, your budget, and what you enjoy. A team of specialised AI agents collaborates to build a complete travel plan — not just a list of tourist spots, but actual hidden gems, real driving routes with petrol bunks and rest stops, live weather forecasts, and hotel options from budget to 5-star.
Example query:

"Plan 5 days in Rajasthan, ₹30,000 budget, I love history and street food, hate crowds"

What you get back:

Day-by-day itinerary with timings
Hidden gems locals actually visit
Real driving route from Google-compatible map links
Petrol bunks and rest stops along the way
Weather forecast for your travel dates
Hotel options with pricing in INR
Full budget breakdown per day


Architecture
User Query
    ↓
FastAPI Gateway
    ↓
Orchestrator Agent   ← coordinates everything
    ↓
┌─────────────────────────────────────┐
│  Discovery  │  Route   │  Budget    │
│  Agent      │  Agent   │  Agent     │
│             │          │            │
│  Finds      │  Routes  │  Tracks    │
│  hidden     │  + stops │  costs +   │
│  gems       │          │  hotels    │
└─────────────────────────────────────┘
    ↓
Critic + Writer Agent   ← validates + final output
    ↓
Structured travel plan

Agent roles and responsibilities
Orchestrator Agent

Receives the user query
Breaks it into sub-tasks
Decides which agents to call and in what order
Assembles the final response from all agents
Handles replanning if any agent fails

Discovery Agent

Finds places matching user preferences
Searches the hidden gems knowledge base (Qdrant)
Filters out tourist traps using review count logic
Returns places with coordinates, timings, and cost estimates

Route Agent

Takes list of places from Discovery Agent
Calculates optimised driving/biking route
Finds petrol bunks and rest stops every 50km
Returns turn-by-turn directions and map links

Budget Agent

Estimates cost per activity, meal, and transport
Tracks running total against user's budget limit
Searches hotels via Booking.com API
Warns when approaching budget limit
Suggests cheaper alternatives

Critic + Writer Agent

Reviews the full plan from all agents
Checks for logical errors (closed venues, impossible timings, back-to-back walking in heat)
Scores the plan and sends back for revision if below threshold
Produces the final formatted travel plan


Memory
TypeStorageWhat it storesShort termStrands sessionCurrent conversation contextEpisodicQdrantPast trips, user preferences across sessionsSemanticQdrantHidden gems knowledge base, city guides

Tools and APIs
ToolPurposeCostNominatim (OpenStreetMap)Place name → coordinatesFree, no keyOpen-MeteoWeather forecastFree, no keyOpenRouteServiceRoutes, petrol bunks, rest stopsFree 2000/dayBooking.com (RapidAPI)Hotel searchFree tierTripAdvisor (RapidAPI)RestaurantsFree tier

Tech stack

Agent framework — Strands Agents
LLM (reasoning) — Gemini Flash 2.0 (free tier)
LLM (fast tasks) — Ollama Qwen3:4b (local)
Vector store — Qdrant (free cloud tier)
API layer — FastAPI
Containerisation — Docker


Repo structure
travelstack/
├── agents/
│   ├── base_agent.py
│   ├── orchestrator.py
│   ├── discovery.py
│   ├── route.py
│   ├── budget.py
│   └── critic_writer.py
├── tools/
│   ├── geocode.py
│   ├── weather.py
│   ├── routes.py
│   └── hotels.py
├── memory/
│   ├── short_term.py
│   ├── episodic.py
│   └── semantic.py
├── guardrails.py
├── observability.py
├── main.py
├── .env.example
├── requirements.txt
└── docker-compose.yml

Guardrails

Budget limit validation before finalising plan
Date validation (no past dates)
Empty results fallback (agent retries with broader search)
Venue opening hours check
Impossible schedule detection (critic agent)


Observability

Every tool call logged with latency
Token usage tracked per agent
Full trace of agent decisions available via /trace endpoint


Setup
bashgit clone https://github.com/yourusername/travelstack
cd travelstack
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # add your API keys
uvicorn main:app --reload

Phase roadmap

 Tool functions (geocoding, weather, routes, hotels)
 Base agent + LLM client
 Discovery agent with Qdrant RAG
 Orchestrator + multi-agent coordination
 Route + Budget agents
 Critic + Writer agent
 Memory integration (episodic + semantic)
 Guardrails + observability
 Eval suite (20 test scenarios)
 Docker deployment


Design decisions
Why free APIs only? All external APIs used are free tier or completely free with no billing required. The entire stack can run at zero cost during development.
Why Strands over LangChain? Strands gives us direct control over the agent loop without the abstraction overhead. Each agent's reasoning is explicit and debuggable.
Why two LLMs? Gemini Flash handles complex multi-step reasoning (planning, synthesis). Ollama Qwen3 handles fast, simple tasks (classification, routing) locally — keeping costs at zero for high-frequency calls.
Why Qdrant for memory? Single vector store handles both episodic memory (past user preferences) and semantic memory (places knowledge base) in separate collections. Clean upgrade path to Qdrant Cloud for production.