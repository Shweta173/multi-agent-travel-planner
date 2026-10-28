import os
from datetime import date
from typing import TypedDict, Annotated, Optional
import operator

import psycopg
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq

from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL is not set. Please check your .env file."
    )


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b"
)


# ============================================================
# STATE
# ============================================================

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int


# ============================================================
# FLIGHT AGENT
# ============================================================

class FlightRoute(BaseModel):
    origin: Optional[str] = Field(
        default=None,
        description="IATA code of the departure airport, or null if the user didn't say where they are travelling from"
    )
    destination: Optional[str] = Field(
        default=None,
        description="IATA code of the main airport for the destination, e.g. GOI for Goa or DPS for Bali"
    )
    flight_date: Optional[str] = Field(
        default=None,
        description="Outbound travel date as YYYY-MM-DD, or null if no specific date was given"
    )


route_llm = llm.with_structured_output(FlightRoute)


def flight_agent(state: TravelState):

    query = state["user_query"]

    route = route_llm.invoke(
        [
            SystemMessage(
                content=(
                    "Extract the flight route from the travel request. "
                    f"Today's date is {date.today().isoformat()}. "
                    "Use 3-letter IATA airport codes."
                )
            ),
            HumanMessage(
                content=query
            )
        ]
    )

    flight_data = search_flights(
        origin=(route.origin or "").upper() or None,
        destination=(route.destination or "").upper() or None,
        flight_date=route.flight_date
    )

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(
                content="Flight results fetched"
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# ============================================================
# HOTEL AGENT
# ============================================================

def hotel_agent(state: TravelState):

    query = f"Best hotels for {state['user_query']}"

    hotel_results = tavily_search(query)

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(
                content="Hotel information fetched"
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# ============================================================
# ITINERARY AGENT
# ============================================================

def itinerary_agent(state: TravelState):

    prompt = f"""
Create a travel itinerary.

User Query:
{state['user_query']}

Flight Results:
{state['flight_results']}

Hotel Results:
{state['hotel_results']}
"""

    response = llm.invoke(
        [
            SystemMessage(
                content="You are an expert travel planner."
            ),
            HumanMessage(
                content=prompt
            )
        ]
    )

    return {
        "itinerary": response.content,
        "messages": [
            response
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# ============================================================
# FINAL RESPONSE AGENT
# ============================================================

def final_agent(state: TravelState):

    final_prompt = f"""
Generate a final travel response for the user.

Flights:
{state['flight_results']}

Hotels:
{state['hotel_results']}

Itinerary:
{state['itinerary']}
"""

    response = llm.invoke(
        [
            HumanMessage(
                content=final_prompt
            )
        ]
    )

    return {
        "messages": [
            response
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# ============================================================
# BUILD GRAPH
# ============================================================

graph = StateGraph(TravelState)


# Add nodes
graph.add_node(
    "flight_agent",
    flight_agent
)

graph.add_node(
    "hotel_agent",
    hotel_agent
)

graph.add_node(
    "itinerary_agent",
    itinerary_agent
)

graph.add_node(
    "final_agent",
    final_agent
)


# ============================================================
# GRAPH FLOW
# ============================================================

graph.add_edge(
    START,
    "flight_agent"
)

graph.add_edge(
    "flight_agent",
    "hotel_agent"
)

graph.add_edge(
    "hotel_agent",
    "itinerary_agent"
)

graph.add_edge(
    "itinerary_agent",
    "final_agent"
)

graph.add_edge(
    "final_agent",
    END
)


# ============================================================
# POSTGRESQL CHECKPOINTER
# ============================================================

# IMPORTANT:
# autocommit=True is required because LangGraph's PostgreSQL
# migrations use CREATE INDEX CONCURRENTLY, which PostgreSQL
# does not allow inside a transaction.

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True
)

checkpointer = PostgresSaver(_conn)


# Create/update LangGraph checkpoint tables
checkpointer.setup()


# Compile graph with persistent PostgreSQL checkpointer
app = graph.compile(
    checkpointer=checkpointer
)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Travel Multi-Agent System Started")
    print("----------------------------------")

    config = {
        "configurable": {
            "thread_id": "user_aarohi"
        }
    }

    user_input = input(
        "Enter travel request: "
    )

    result = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content=user_input
                )
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )

    print("\nFINAL RESPONSE:\n")
    print("----------------------------------")

    # Print final AI response
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(msg.content)