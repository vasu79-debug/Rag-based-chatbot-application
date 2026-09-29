import json
import logging
from datetime import datetime, date
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("Subzillo CRUD Server")

db_available = False
try:
    from database import SessionLocal, Subscription
    db_available = True
    logger.info("Successfully connected to Postgres 'mcpchat' database.")
except Exception as e:
    logger.error(f"Failed to connect to Database: {e}")

@mcp.tool()
def get_subscriptions() -> str:
    """Retrieve all current user subscriptions."""
    if not db_available:
        return json.dumps({"error": "Database not available."})
    
    db = SessionLocal()
    try:
        subs = db.query(Subscription).all()
        results = []
        for s in subs:
            results.append({
                "id": s.id,
                "service_name": s.service_name,
                "cost": s.cost,
                "billing_cycle": s.billing_cycle,
                "next_payment_date": str(s.next_payment_date),
                "category": s.category,
                "notes": s.notes
            })
        return json.dumps({"status": "success", "data": results})
    finally:
        db.close()

@mcp.tool()
def create_subscription(
    service_name: str, 
    cost: float, 
    billing_cycle: str, 
    next_payment_date: str, 
    category: str = "", 
    notes: str = ""
) -> str:
    """Create a new subscription."""
    if not db_available:
        return json.dumps({"error": "Database not available."})
    
    db = SessionLocal()
    try:
        date_obj = datetime.strptime(next_payment_date, "%Y-%m-%d").date()
        new_sub = Subscription(
            service_name=service_name,
            cost=cost,
            billing_cycle=billing_cycle,
            next_payment_date=date_obj,
            category=category,
            notes=notes
        )
        db.add(new_sub)
        db.commit()
        db.refresh(new_sub)
        return json.dumps({"status": "success", "message": f"Added {service_name}", "id": new_sub.id})
    except Exception as e:
        db.rollback()
        return json.dumps({"error": str(e)})
    finally:
        db.close()

@mcp.tool()
def delete_subscription(service_name: str) -> str:
    """Cancel or delete a subscription by its service name."""
    if not db_available:
        return json.dumps({"error": "Database not available."})
    
    db = SessionLocal()
    try:
        sub = db.query(Subscription).filter(Subscription.service_name.ilike(f"%{service_name}%")).first()
        if not sub:
            return json.dumps({"error": f"Subscription for {service_name} not found."})
        db.delete(sub)
        db.commit()
        return json.dumps({"status": "success", "message": f"Successfully cancelled {service_name}."})
    finally:
        db.close()

@mcp.tool()
def get_alternative_plans(service_name: str) -> str:
    """Retrieve cheaper alternative plans or competitors for a given subscription service to save money."""
    service = service_name.lower()
    
    alternatives = {
        "netflix": [
            {"plan": "Basic with Ads", "cost": 99, "billing_cycle": "monthly", "features": "720p, Ad-supported"},
            {"plan": "Mobile Only", "cost": 149, "billing_cycle": "monthly", "features": "480p, Mobile/Tablet only"}
        ],
        "spotify": [
            {"plan": "Spotify Student", "cost": 59, "billing_cycle": "monthly", "features": "Ad-free, Student ID required"},
            {"plan": "Spotify Mini", "cost": 25, "billing_cycle": "weekly", "features": "Mobile only, Ad-free"}
        ],
        "adobe creative cloud": [
            {"plan": "Photography Plan", "cost": 799, "billing_cycle": "monthly", "features": "Photoshop & Lightroom only"},
            {"plan": "Canva Pro (Competitor)", "cost": 399, "billing_cycle": "monthly", "features": "Graphic design alternative"}
        ],
        "amazon prime": [
            {"plan": "Prime Lite", "cost": 799, "billing_cycle": "yearly", "features": "SD Video, 2-day delivery"}
        ]
    }
    
    for key in alternatives:
        if key in service:
            return json.dumps({"status": "success", "service": service_name, "alternatives": alternatives[key]})
            
    return json.dumps({"status": "success", "service": service_name, "alternatives": [], "message": "No cheaper alternatives found."})

@mcp.tool()
def get_usage_statistics(service_name: str) -> str:
    """Check how often the user has actually used a subscription service in the last 30 days."""
    service = service_name.lower()
    
    # Mock data showing high, medium, and low usage
    if "netflix" in service:
        return json.dumps({"service": service_name, "usage_last_30_days": "45 hours", "status": "Active (High Usage)"})
    elif "spotify" in service:
        return json.dumps({"service": service_name, "usage_last_30_days": "120 hours", "status": "Active (High Usage)"})
    elif "adobe" in service:
        return json.dumps({"service": service_name, "usage_last_30_days": "0 hours", "status": "Inactive (Not used in 3 months)"})
    elif "gym" in service or "fitness" in service:
        return json.dumps({"service": service_name, "usage_last_30_days": "1 visit", "status": "Low Usage (Wasted money)"})
    
    return json.dumps({"service": service_name, "usage_last_30_days": "Unknown", "status": "Moderate Usage"})

@mcp.tool()
def search_public_subscription_data(query: str) -> str:
    """Use this tool to search the live internet for public pricing, plans, and features of a subscription service."""
    try:
        # pyrefly: ignore [missing-import]
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query + " subscription pricing plans India rupees", max_results=3))
            
            if not results:
                return json.dumps({"status": "error", "message": f"No public data found for '{query}'."})
                
            formatted_results = []
            for r in results:
                formatted_results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "source": r.get("href", "")
                })
                
            return json.dumps({
                "status": "success", 
                "query": query,
                "search_results": formatted_results
            })
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Failed to search the web: {str(e)}"})

if __name__ == "__main__":
    # mcp.run(transport="stdio")
    print("Starting MCP Server on http://127.0.0.1:8001/sse")
    mcp.run(transport="sse", host="127.0.0.1", port=8001)
