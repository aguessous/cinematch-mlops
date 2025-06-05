from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mlflow.pyfunc
import pandas as pd
import os
from datetime import datetime
import logging
from sqlalchemy import create_engine, text
from pydantic import BaseModel
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CineMatch API",
    description="Système de recommandation de films",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèle global
model = None
engine = None

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[dict]
    model_version: Optional[str] = None
    timestamp: str

@app.on_event("startup")
async def startup_event():
    global model, engine
    
    # Connexion DB
    postgres_uri = os.getenv('POSTGRES_URI')
    if postgres_uri:
        engine = create_engine(postgres_uri)
        logger.info("Connexion PostgreSQL établie")
    
    # Chargement du modèle
    try:
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
        mlflow.set_tracking_uri(mlflow_uri)
        
        model = mlflow.pyfunc.load_model("models:/cinematch-recommender/Production")
        logger.info("Modèle chargé depuis MLflow")
    except Exception as e:
        logger.error(f"Erreur chargement modèle: {e}")
        model = None

@app.get("/")
async def root():
    return {
        "message": "CineMatch API", 
        "status": "active",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "loaded" if model else "not_loaded",
        "database": "connected" if engine else "not_connected"
    }

@app.get("/recommend/{user_id}")
async def get_recommendations(user_id: int, n: int = 10):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    
    try:
        # Prédiction
        input_data = pd.DataFrame({
            'user_id': [user_id],
            'n_recommendations': [n]
        })
        
        result = model.predict(input_data)
        
        # Enrichissement avec les titres de films
        recommendations = result['recommendations']
        
        if engine and recommendations:
            # Récupération des informations films
            movie_ids = [rec['movie_id'] for rec in recommendations]
            query = text("""
                SELECT movie_id, title, genres 
                FROM movies 
                WHERE movie_id = ANY(:movie_ids)
            """)
            
            movies_df = pd.read_sql(query, engine, params={'movie_ids': movie_ids})
            movies_dict = movies_df.set_index('movie_id').to_dict('index')
            
            # Enrichissement
            for rec in recommendations:
                movie_info = movies_dict.get(rec['movie_id'], {})
                rec['title'] = movie_info.get('title', 'Titre inconnu')
                rec['genres'] = movie_info.get('genres', 'Non spécifié')
        
        # Log des prédictions
        if engine:
            log_predictions(user_id, recommendations)
        
        response = RecommendationResponse(
            user_id=user_id,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Erreur recommandation pour user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def log_predictions(user_id: int, recommendations: List[dict]):
    """Log des prédictions en base pour monitoring"""
    try:
        timestamp = datetime.now()
        
        for rec in recommendations:
            query = text("""
                INSERT INTO predictions (timestamp, user_id, movie_id, score, model_version)
                VALUES (:timestamp, :user_id, :movie_id, :score, :model_version)
            """)
            
            engine.execute(query, {
                'timestamp': timestamp,
                'user_id': user_id,
                'movie_id': rec['movie_id'],
                'score': rec['score'],
                'model_version': 'v1.0'
            })
            
    except Exception as e:
        logger.error(f"Erreur logging prédictions: {e}")

@app.get("/stats")
async def get_stats():
    """Statistiques de l'API"""
    if not engine:
        raise HTTPException(status_code=503, detail="Base de données non disponible")
    
    try:
        # Statistiques générales
        stats_query = text("""
            SELECT 
                COUNT(*) as total_predictions,
                COUNT(DISTINCT user_id) as unique_users,
                COUNT(DISTINCT movie_id) as unique_movies,
                AVG(score) as avg_score
            FROM predictions
            WHERE timestamp > NOW() - INTERVAL '24 hours'
        """)
        
        result = engine.execute(stats_query).fetchone()
        
        return {
            "last_24h": {
                "total_predictions": result[0],
                "unique_users": result[1], 
                "unique_movies": result[2],
                "avg_score": float(result[3]) if result[3] else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Erreur stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)