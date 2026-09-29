import os
from sqlalchemy import create_engine, Column, Integer, String, Float, Date
from sqlalchemy.orm import declarative_base, sessionmaker

# Normally this comes from .env, using a hardcoded fallback for local dev
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg2://postgres:Krify%40123@localhost:5432/postgres")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    service_name = Column(String, index=True, nullable=False)
    cost = Column(Float, nullable=False)
    billing_cycle = Column(String, nullable=False)  # 'monthly', 'yearly', 'weekly'
    next_payment_date = Column(Date, nullable=False)
    category = Column(String, nullable=True)
    notes = Column(String, nullable=True)

# Create tables automatically
Base.metadata.create_all(bind=engine)
