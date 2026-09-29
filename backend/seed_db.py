import os
import sys
from datetime import date, timedelta
from database import SessionLocal, Subscription

def seed_db():
    db = SessionLocal()
    
    # Check if there are already subscriptions
    if db.query(Subscription).count() > 0:
        print("Database already has subscriptions. Skipping seed.")
        db.close()
        return

    print("Seeding dummy subscriptions...")
    
    dummy_subs = [
        Subscription(
            service_name="Netflix", 
            cost=15.99, 
            billing_cycle="monthly", 
            next_payment_date=date.today() + timedelta(days=10),
            category="Entertainment",
            notes="Standard HD Plan"
        ),
        Subscription(
            service_name="Amazon Prime", 
            cost=139.00, 
            billing_cycle="yearly", 
            next_payment_date=date.today() + timedelta(days=45),
            category="Shopping",
            notes="Includes Prime Video"
        ),
        Subscription(
            service_name="Spotify", 
            cost=10.99, 
            billing_cycle="monthly", 
            next_payment_date=date.today() + timedelta(days=5),
            category="Entertainment",
            notes="Premium Individual"
        ),
        Subscription(
            service_name="Gym Membership", 
            cost=45.00, 
            billing_cycle="monthly", 
            next_payment_date=date.today() + timedelta(days=2),
            category="Health",
            notes="Local Fitness Center"
        )
    ]
    
    for sub in dummy_subs:
        db.add(sub)
        
    db.commit()
    print("Successfully added 4 dummy subscriptions to Postgres!")
    db.close()

if __name__ == "__main__":
    seed_db()
