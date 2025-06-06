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

MODEL_NAME = "cinematch-recommender"
MODEL_ALIAS = "champion"

from mlflow.exceptions import MlflowException   # ← import en plus

def load_model_from_registry():
    """
    Tente de charger le modèle cinematch‑recommender@champion.
    Renvoie l'objet modèle ou None s'il n'existe pas encore.
    """
    try:
        uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        logger.info(f"Tentative de chargement via l'alias : {uri}")
        return mlflow.pyfunc.load_model(uri)
    except MlflowException as e:
        logger.warning(f"Alias '{MODEL_ALIAS}' indisponible ({e_alias})")

        try:
            uri = f"models:/{MODEL_NAME}/Production"
            logger.info(f"Tentative de chargement via le stage : {uri}")
            return mlflow.pyfunc.load_model(uri)

        except (MlflowException, Exception) as e_stage:
            logger.warning(f"Aucune version en Production ({e_stage})")
            return None


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
        
        model = load_model_from_registry()          # ← nouvelle fonction
        if model:
            logger.info("Modèle chargé depuis MLflow")
        else:
            logger.warning("Pas encore de modèle disponible.")
        
        # model = mlflow.pyfunc.load_model("models:/cinematch-recommender@champion")
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
    global model
    if model is None:                          # 1re tentative avion
        model = load_model_from_registry()     # ← recharge à chaud
    if model is None:
        raise HTTPException(503, detail="Modèle non disponible")
    
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

from sqlalchemy import text

def log_predictions(user_id: int, recommendations: List[dict]):
    """Log des prédictions en base pour monitoring"""
    if not engine or not recommendations:
        return

    timestamp = datetime.now()

    insert_sql = text("""
        INSERT INTO predictions (timestamp, user_id, movie_id, score, model_version)
        VALUES (:timestamp, :user_id, :movie_id, :score, :model_version)
    """)

    with engine.begin() as conn:              # ouverture + commit implicite
        for rec in recommendations:
            conn.execute(
                insert_sql,
                {
                    "timestamp": timestamp,
                    "user_id": user_id,
                    "movie_id": rec["movie_id"],
                    "score": rec["score"],
                    "model_version": "v13"    # ou la version courante
                },
            )


from sqlalchemy import text
from sqlalchemy.engine import Row

@app.get("/stats")
async def get_stats():
    """Statistiques agrégées sur les 24 dernières heures."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Base de données non disponible")

    stats_sql = text("""
        SELECT 
            COUNT(*)                     AS total_predictions,
            COUNT(DISTINCT user_id)      AS unique_users,
            COUNT(DISTINCT movie_id)     AS unique_movies,
            AVG(score)                   AS avg_score
        FROM predictions
        WHERE timestamp > NOW() - INTERVAL '24 hours'
    """)

    try:
        with engine.connect() as conn:               # ← plus d'engine.execute
            row: Row = conn.execute(stats_sql).one()

        return {
            "last_24h": {
                "total_predictions": row.total_predictions,
                "unique_users":     row.unique_users,
                "unique_movies":    row.unique_movies,
                "avg_score":        float(row.avg_score or 0)
            }
        }

    except Exception as e:
        logger.error(f"Erreur stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)