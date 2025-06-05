import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import implicit
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix
import mlflow
import mlflow.pyfunc
import pickle
import os
import psycopg2
from sqlalchemy import create_engine
import logging
import warnings

# Ignorer les warnings spécifiques
warnings.filterwarnings("ignore", category=UserWarning, module='implicit')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Solution de contournement pour l'erreur de configuration BLAS
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["IMPLICIT_NO_CHECK_BLAS"] = "1"  # Désactive la vérification BLAS

class MovieRecommender(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.model = None
        self.user_item_matrix = None
        self.user_mapping = None
        self.item_mapping = None
        self.reverse_item_mapping = None
        
    def load_context(self, context):
        # Le modèle sera chargé lors de l'initialisation
        pass
    
    def predict(self, context, model_input):
        user_id = model_input['user_id'][0]
        n_recommendations = model_input.get('n_recommendations', [10])[0]
        
        if user_id not in self.user_mapping:
            return {'recommendations': []}
            
        user_idx = self.user_mapping[user_id]
        recommendations = self.model.recommend(
            useridx=user_idx, 
            user_items=self.user_item_matrix[user_idx], 
            N=n_recommendations,
            filter_already_liked_items=True
        )
        
        results = []
        for item_idx, score in zip(recommendations[0], recommendations[1]):
            movie_id = self.reverse_item_mapping[item_idx]
            results.append({
                'movie_id': int(movie_id),
                'score': float(score)
            })
            
        return {'user_id': user_id, 'recommendations': results}

def load_data():
    """Charge les données MovieLens depuis PostgreSQL ou CSV"""
    postgres_uri = os.getenv('POSTGRES_URI')
    
    if postgres_uri:
        logger.info("Chargement depuis PostgreSQL...")
        engine = create_engine(postgres_uri)
        ratings = pd.read_sql('SELECT * FROM ratings', engine)
        movies = pd.read_sql('SELECT * FROM movies', engine)
    else:
        logger.info("Chargement depuis CSV...")
        ratings = pd.read_csv('data/raw/ratings.dat', 
                            sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'],
                            engine='python')
        movies = pd.read_csv('data/raw/movies.dat',
                           sep='::', names=['movie_id', 'title', 'genres'],
                           engine='python', encoding='latin1')
    
    return ratings, movies
    

def create_user_item_matrix(ratings):
    """Crée la matrice user-item sparse"""
    # Mapping des IDs
    user_ids = ratings['user_id'].unique()
    movie_ids = ratings['movie_id'].unique()
    
    user_mapping = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_mapping = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    reverse_item_mapping = {idx: movie_id for movie_id, idx in item_mapping.items()}
    
    # Création de la matrice sparse
    user_indices = [user_mapping[user_id] for user_id in ratings['user_id']]
    item_indices = [item_mapping[movie_id] for movie_id in ratings['movie_id']]
    
    user_item_matrix = coo_matrix(
        (ratings['rating'], (user_indices, item_indices)),
        shape=(len(user_ids), len(movie_ids))
    ).tocsr()
    
    return user_item_matrix, user_mapping, item_mapping, reverse_item_mapping

def train_model():
    """Entraîne le modèle ALS et l'enregistre dans MLflow"""
    mlflow.set_experiment("cinematch-recommendations")
    
    with mlflow.start_run():
        # Chargement des données
        ratings, movies = load_data()
        logger.info(f"Données chargées: {len(ratings)} ratings, {len(movies)} films")
        
        # Préparation de la matrice
        user_item_matrix, user_mapping, item_mapping, reverse_item_mapping = create_user_item_matrix(ratings)
        
        # Hyperparamètres
        factors = 50
        regularization = 0.01
        iterations = 20
        
        # Log des paramètres
        mlflow.log_param("factors", factors)
        mlflow.log_param("regularization", regularization)
        mlflow.log_param("iterations", iterations)
        mlflow.log_param("n_users", len(user_mapping))
        mlflow.log_param("n_items", len(item_mapping))
        
        # Entraînement
        model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=42,
            calculate_training_loss=True,
            use_gpu=False  # Désactiver GPU pour éviter d'autres problèmes
        )
        
        logger.info("Début de l'entraînement...")
        model.fit(user_item_matrix)
        logger.info("Entraînement terminé")
        
        # Évaluation simple
        train_matrix, test_matrix = train_test_split(ratings, test_size=0.2, random_state=42)
        
        # Métriques basiques
        n_recommendations = 10
        precision_scores = []
        
        # Échantillonner seulement 100 utilisateurs pour l'évaluation
        test_users = test_matrix['user_id'].unique()[:100]
        test_subset = test_matrix[test_matrix['user_id'].isin(test_users)]
        
        for user_id in test_users:
            if user_id in user_mapping:
                user_idx = user_mapping[user_id]
                recommendations = model.recommend(
                    useridx=user_idx, 
                    user_items=user_item_matrix[user_idx], 
                    N=n_recommendations
                )
                
                recommended_items = set([reverse_item_mapping[item_idx] for item_idx in recommendations[0]])
                actual_items = set(test_subset[test_subset['user_id'] == user_id]['movie_id'])
                
                if len(recommended_items) > 0:
                    precision = len(recommended_items.intersection(actual_items)) / len(recommended_items)
                    precision_scores.append(precision)
        
        avg_precision = np.mean(precision_scores) if precision_scores else 0
        mlflow.log_metric("precision_at_10", avg_precision)
        
        # Création de l'objet modèle
        recommender = MovieRecommender()
        recommender.model = model
        recommender.user_item_matrix = user_item_matrix
        recommender.user_mapping = user_mapping
        recommender.item_mapping = item_mapping
        recommender.reverse_item_mapping = reverse_item_mapping
        
        # Enregistrement du modèle
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=recommender,
            conda_env={
                'name': 'cinematch-env',
                'channels': ['defaults', 'conda-forge'],
                'dependencies': [
                    'python=3.9',
                    'pip',
                    {'pip': [
                        'implicit==0.6.2',
                        'scikit-learn==1.3.0',
                        'pandas==2.0.3',
                        'numpy==1.24.3'
                    ]}
                ]
            }
        )
        
        # Promotion automatique en Production si précision > seuil
        if avg_precision > 0.05:  # Seuil à ajuster
            client = mlflow.MlflowClient()
            model_version = client.create_model_version(
                name="cinematch-recommender",
                source=mlflow.get_artifact_uri("model"),
                run_id=mlflow.active_run().info.run_id
            )
            
            client.transition_model_version_stage(
                name="cinematch-recommender",
                version=model_version.version,
                stage="Production"
            )
            
            logger.info(f"Modèle promu en Production (version {model_version.version})")
        
        logger.info(f"Précision moyenne: {avg_precision:.4f}")

if __name__ == "__main__":
    train_model()