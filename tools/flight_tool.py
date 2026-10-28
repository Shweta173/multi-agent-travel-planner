import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AVIATIONSTACK_API_KEY")

URL = "http://api.aviationstack.com/v1/flights"


def _format_flight(flight):

    airline = (flight.get("airline") or {}).get("name") or "Unknown"

    flight_number = (flight.get("flight") or {}).get("iata") or "Unknown"

    departure = flight.get("departure") or {}
    arrival = flight.get("arrival") or {}

    return f"""
Airline: {airline}
Flight: {flight_number}
Departure: {departure.get("airport") or "Unknown"} ({departure.get("iata") or "?"}) at {departure.get("scheduled") or "Unknown"}
Arrival: {arrival.get("airport") or "Unknown"} ({arrival.get("iata") or "?"}) at {arrival.get("scheduled") or "Unknown"}
Status: {flight.get("flight_status") or "Unknown"}
"""


def search_flights(origin=None, destination=None, flight_date=None):
    """
    Search flights between two airports.

    origin / destination are IATA airport codes (e.g. "BOM", "GOI").
    flight_date is "YYYY-MM-DD" and is optional.
    """

    if not destination:
        return "No destination airport could be identified in the request, so no flights were searched."

    params = {
        "access_key": API_KEY,
        "arr_iata": destination,
        "limit": 5
    }

    if origin:
        params["dep_iata"] = origin

    note = ""

    if flight_date:
        params["flight_date"] = flight_date

    try:
        data = requests.get(URL, params=params, timeout=20).json()

        # The free Aviationstack plan doesn't allow date filters,
        # so retry without the date and say so in the results.
        if flight_date and (data.get("error") or {}).get("code") == "function_access_restricted":
            params.pop("flight_date")
            note = f"Note: flights for {flight_date} need a paid Aviationstack plan, so these are current flights on this route.\n"
            data = requests.get(URL, params=params, timeout=20).json()

    except requests.RequestException as e:
        return f"Flight search failed: {e}"

    if "error" in data:
        message = (data.get("error") or {}).get("message", "Unknown error")
        return f"Flight search failed: {message}"

    route = f"{origin or 'any airport'} → {destination}"

    flights = [_format_flight(f) for f in data.get("data", [])[:5]]

    if not flights:
        return f"{note}No flights found for {route}."

    return f"{note}Flights for {route}:\n" + "\n".join(flights)
