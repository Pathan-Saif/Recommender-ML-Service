from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    item_id = Column(String, index=True)
    event_type = Column(String)          # view, click, purchase
    weight = Column(Float)               # numeric score
    timestamp = Column(DateTime)
