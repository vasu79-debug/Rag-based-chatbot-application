from mcp.server.fastmcp import FastMCP
import json
from rag.hybrid_retriever import hybrid_retriever

# Initialize FastMCP Server
mcp = FastMCP("krify_demo5_tools_server")

# --- Dummy APIs wrapped as MCP tools ---

@mcp.tool()
def get_appointment_details(date: str) -> str:
    """Get the appointment details for a specific date in YYYY-MM-DD or 'tomorrow', 'today' formats."""
    # Simulate a missing record
    if date.lower() == "yesterday":
        return json.dumps({"error": "No records found for yesterday."})
    
    # Simulate an API error / malformed data
    if date == "error":
        return "<html><body>500 Internal Server Error</body></html>"
        
    return json.dumps({
        "status": "success",
        "data": {
            "date": date,
            "time": "11:30 AM",
            "doctor": "Dr. Menon",
            "department": "Oncology",
            "location": "2nd floor"
        }
    })

@mcp.tool()
def get_leave_balance(employee_id: str) -> str:
    """Get the leave balance for an employee. Pass the employee ID."""
    # Simulate permission enforcement based on dummy token checking
    if employee_id != "EMP123":
        return json.dumps({"error": "Permission denied: You can only access your own leave balance."})
        
    return json.dumps({
        "status": "success",
        "data": {
            "employee_id": employee_id,
            "annual_leave": 14,
            "sick_leave": 5
        }
    })

import requests

@mcp.tool()
def get_weather(location: str) -> str:
    """Get the current weather for a specific city location (e.g., 'London', 'New York', 'Hyderabad')."""
    try:
        # 1. Geocoding: Get latitude and longitude for the location
        geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
        geo_response = requests.get(geocode_url)
        geo_response.raise_for_status()
        geo_data = geo_response.json()
        
        if not geo_data.get("results"):
            return json.dumps({"error": f"Could not find coordinates for location: {location}"})
            
        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]
        resolved_name = geo_data["results"][0]["name"]
        country = geo_data["results"][0].get("country", "")
        
        # 2. Fetch current weather
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_response = requests.get(weather_url)
        weather_response.raise_for_status()
        weather_data = weather_response.json()
        
        current = weather_data.get("current_weather", {})
        
        return json.dumps({
            "status": "success",
            "data": {
                "location_requested": location,
                "location_resolved": f"{resolved_name}, {country}",
                "temperature": f"{current.get('temperature')}°C",
                "windspeed": f"{current.get('windspeed')} km/h",
                "is_day": bool(current.get('is_day'))
            }
        })
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch weather data: {str(e)}"})

@mcp.tool()
def search_krify_knowledge(query: str) -> str:
    """Search the Krify organizational knowledge base for information about Krify, company policies, and services. Use this when the user asks questions related to Krify."""
    try:
        # We use the hybrid retriever to fetch chunks from ChromaDB & BM25
        results = hybrid_retriever.retrieve(query)
        if not results:
            return "No information found in the Krify knowledge base."
            
        formatted_results = []
        for i, chunk in enumerate(results[:3], start=1):
            source = chunk.metadata.get("source", "Document")
            page = chunk.metadata.get("page", 1)
            formatted_results.append(f"Source [{i}]: {source} (Page {page})\n{chunk.page_content}\n")
            
        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"

if __name__ == "__main__":
    # Start the server using stdio transport
    mcp.run()
