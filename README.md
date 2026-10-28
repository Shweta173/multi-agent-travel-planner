# AI Travel Planner (Multi-Agent, LangGraph)

A multi-agent travel planning app built with LangGraph. You describe a trip in plain language, such as "plan a trip to Bali with 40k in Dec 25-31". A pipeline of agents then gathers flight data, searches for hotels, writes a day-by-day itinerary and produces a final travel plan. It has a Streamlit web UI, and conversation state is saved in PostgreSQL.

## How it works

```
START → Flight Agent → Hotel Agent → Itinerary Agent → Final Agent → END
```

| Agent | What it does | Source |
| --- | --- | --- |
| Flight Agent | Fetches live flight data | Aviationstack API |
| Hotel Agent | Searches the web for the best hotels for the trip | Tavily Search API |
| Itinerary Agent | Writes a travel itinerary from the query, flights and hotels | Groq LLM (`openai/gpt-oss-120b`) |
| Final Agent | Combines everything into the final travel response | Groq LLM |

All agents share a typed state (`TravelState`) holding the messages, user query, flight results, hotel results, itinerary and an LLM call counter. The graph is compiled with a **PostgreSQL checkpointer** (`PostgresSaver`), so each user ID (thread) keeps its history across runs.

## Features

- **Streamlit UI:** describe your trip or pick a quick-start example (Japan, Paris, Dubai, Bali).
- **Live agent pipeline:** each agent's output (flights, hotels, itinerary, final plan) is shown as it completes.
- **Persistent sessions:** a User ID field maps to a LangGraph thread, and its state is stored in PostgreSQL.
- **Save and download:** every generated plan is auto-saved as Markdown in `travel_plans/` and can be downloaded from the UI.
- **CLI mode:** run `main.py` directly to plan a trip from the terminal.

## Tech stack

- **Python 3.13**
- **LangGraph** for the multi-agent graph and state management
- **LangChain** + **Groq** for the LLM calls
- **PostgreSQL** + `langgraph-checkpoint-postgres` for persistent checkpoints
- **Tavily** for hotel web search
- **Aviationstack** for flight data
- **Streamlit** for the web interface

## Project structure

```
├── main.py              # Agents, LangGraph graph, PostgreSQL checkpointer, CLI entry point
├── frontend.py          # Streamlit web app
├── tools/
│   ├── flight_tool.py   # Aviationstack flight lookup
│   └── tavily_tool.py   # Tavily hotel search
├── requirements.txt
└── .env.example
```

## Running it locally

### Prerequisites
- Python 3.13
- A running PostgreSQL database
- API keys for [Groq](https://console.groq.com/keys), [Tavily](https://app.tavily.com) and [Aviationstack](https://aviationstack.com)

### Setup

```bash
git clone https://github.com/Shweta173/multi-agent-travel-planner.git
cd multi-agent-travel-planner
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

On macOS or Linux, activate with `source venv/bin/activate` and use `cp` instead of `copy`.

Fill in your API keys and `DATABASE_URL` in `.env`. The LangGraph checkpoint tables are created automatically on the first run.

### Run the web app

```bash
streamlit run frontend.py
```

### Or run it in the terminal

```bash
python main.py
```

## Known limitations

- The flight tool currently returns the latest flights from Aviationstack and doesn't yet filter by the origin, destination or dates in your query. Hotel search and the itinerary do use your query.
- The agents run one after another in a fixed order. Flights and hotels could run in parallel since they don't depend on each other.

## Author

**Shweta Kamble** · [GitHub](https://github.com/Shweta173)
