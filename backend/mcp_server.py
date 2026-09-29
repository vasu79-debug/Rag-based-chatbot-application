import json
import os
import requests
from mcp.server.fastmcp import FastMCP
from datetime import datetime

# Initialize FastMCP Server
mcp = FastMCP("subzillo_mcp_server")

# Try to import DB components
try:
    from database import SessionLocal, Subscription
    db_available = True
except Exception as e:
    db_available = False
    print(f"Database not configured properly: {e}")

# --- Subzillo Subscription CRUD Tools ---

@mcp.tool()
def get_subscriptions() -> str:
    """Retrieve all active subscriptions for the user."""
    if not db_available:
        return json.dumps({"error": "Database connection failed. Please ensure PostgreSQL is running and DATABASE_URL is set."})
        
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
    except Exception as e:
        return json.dumps({"error": str(e)})
    finally:
        db.close()

@mcp.tool()
def create_subscription(service_name: str, cost: float, billing_cycle: str, next_payment_date: str, category: str = None, notes: str = None) -> str:
    """Add a new subscription. Cost must be > 0. Cycle must be monthly, yearly, or weekly. Date format: YYYY-MM-DD."""
    if cost <= 0:
        return json.dumps({"error": "Cost must be greater than 0."})
    if billing_cycle.lower() not in ["monthly", "yearly", "weekly"]:
        return json.dumps({"error": "Billing cycle must be monthly, yearly, or weekly."})
        
    db = SessionLocal()
    try:
        payment_date = datetime.strptime(next_payment_date, "%Y-%m-%d").date()
        new_sub = Subscription(
            service_name=service_name,
            cost=cost,
            billing_cycle=billing_cycle.lower(),
            next_payment_date=payment_date,
            category=category,
            notes=notes
        )
        db.add(new_sub)
        db.commit()
        db.refresh(new_sub)
        return json.dumps({"status": "success", "message": f"Created subscription {service_name} with ID {new_sub.id}"})
    except ValueError:
        return json.dumps({"error": "Invalid date format. Please use YYYY-MM-DD."})
    except Exception as e:
        db.rollback()
        return json.dumps({"error": str(e)})
    finally:
        db.close()

@mcp.tool()
def update_subscription(sub_id: int, service_name: str = None, cost: float = None, billing_cycle: str = None, next_payment_date: str = None, category: str = None, notes: str = None) -> str:
    """Update an existing subscription by ID. Only provided fields will be updated."""
    db = SessionLocal()
    try:
        sub = db.query(Subscription).filter(Subscription.id == sub_id).first()
        if not sub:
            return json.dumps({"error": f"Subscription with ID {sub_id} not found."})
            
        if service_name:
            sub.service_name = service_name
        if cost is not None:
            if cost <= 0:
                return json.dumps({"error": "Cost must be greater than 0."})
            sub.cost = cost
        if billing_cycle:
            if billing_cycle.lower() not in ["monthly", "yearly", "weekly"]:
                return json.dumps({"error": "Billing cycle must be monthly, yearly, or weekly."})
            sub.billing_cycle = billing_cycle.lower()
        if next_payment_date:
            sub.next_payment_date = datetime.strptime(next_payment_date, "%Y-%m-%d").date()
        if category is not None:
            sub.category = category
        if notes is not None:
            sub.notes = notes
            
        db.commit()
        return json.dumps({"status": "success", "message": f"Updated subscription {sub_id}"})
    except ValueError:
        return json.dumps({"error": "Invalid date format. Please use YYYY-MM-DD."})
    except Exception as e:
        db.rollback()
        return json.dumps({"error": str(e)})
    finally:
        db.close()

@mcp.tool()
def delete_subscription(sub_id: int) -> str:
    """Delete a subscription permanently by ID."""
    db = SessionLocal()
    try:
        sub = db.query(Subscription).filter(Subscription.id == sub_id).first()
        if not sub:
            return json.dumps({"error": f"Subscription with ID {sub_id} not found."})
            
        db.delete(sub)
        db.commit()
        return json.dumps({"status": "success", "message": f"Deleted subscription {sub_id}"})
    except Exception as e:
        db.rollback()
        return json.dumps({"error": str(e)})
    finally:
        db.close()

if __name__ == "__main__":
    # Start the server using stdio transport
    mcp.run()
