from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime
import pandas as pd

from database import get_db, engine
from models import Base, Interaction
from recommender import RecommenderEngine
from utils import map_event_to_weight

app = FastAPI(title="ML Recommender")
Base.metadata.create_all(bind=engine)
recommender = RecommenderEngine()

class InteractionIn(BaseModel):
    userId: int
    externalItemId: str
    eventType: str

@app.get("/health")
def health(): return {"status": "ml-service is running"}

@app.post("/interaction")
def save_interaction(data: InteractionIn, db: Session = Depends(get_db)):
    interaction = Interaction(
        user_id=data.userId,
        item_id=data.externalItemId,
        event_type=data.eventType,
        weight=map_event_to_weight(data.eventType),
        timestamp=datetime.utcnow()
    )
    db.add(interaction)
    db.commit()
    return {"status": "saved"}

@app.post("/train")
def train_model(db: Session = Depends(get_db)):
    interactions = db.query(Interaction).all()
    if not interactions: return {"message": "no data"}

    df = pd.DataFrame([{"user_id": i.user_id, "item_id": i.item_id, "weight": i.weight} for i in interactions])
    recommender.train(df)
    return {"status": "model_trained", "total_interactions": len(df)}

@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = 10):
    return {"recommendations": recommender.recommend(user_id, top_k=k)}
