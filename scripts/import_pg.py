#!/usr/bin/env python3
"""
Import MovieLens 1M into PostgreSQL for CineMatch
------------------------------------------------
• crée / vérifie les tables (users, movies, ratings, predictions)
• insère les données provenant de data/raw/*.dat
"""

import os
import argparse
import logging

import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 1.  Création des tables                                                     #
# --------------------------------------------------------------------------- #
def create_tables(engine) -> None:
    """Crée les tables nécessaires (idempotent)."""

    tables_sql = """
    -- Users
    CREATE TABLE IF NOT EXISTS users (
        user_id     INTEGER PRIMARY KEY,
        gender      CHAR(1),
        age         INTEGER,
        occupation  INTEGER,
        zip_code    VARCHAR(10)
    );

    -- Movies
    CREATE TABLE IF NOT EXISTS movies (
        movie_id    INTEGER PRIMARY KEY,
        title       VARCHAR(255),
        genres      VARCHAR(255)
    );

    -- Ratings
    CREATE TABLE IF NOT EXISTS ratings (
        user_id INTEGER,
        movie_id INTEGER,
        rating  INTEGER,
        timestamp INTEGER,
        PRIMARY KEY (user_id, movie_id),
        FOREIGN KEY (user_id)  REFERENCES users(user_id),
        FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
    );

    -- Predictions (monitoring)
    CREATE TABLE IF NOT EXISTS predictions (
        id            SERIAL PRIMARY KEY,
        timestamp     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        user_id       INTEGER,
        movie_id      INTEGER,
        score         FLOAT,
        model_version VARCHAR(50)
    );

    -- Index
    CREATE INDEX IF NOT EXISTS idx_ratings_user   ON ratings(user_id);
    CREATE INDEX IF NOT EXISTS idx_ratings_movie  ON ratings(movie_id);
    CREATE INDEX IF NOT EXISTS idx_pred_timestamp ON predictions(timestamp);
    """

    with engine.begin() as conn:          # connexion + transaction auto‑commit
        conn.exec_driver_sql(tables_sql)  # pas besoin de `text()` ici
    logger.info("✅ Tables créées / vérifiées")


# --------------------------------------------------------------------------- #
# 2.  Import des CSV / DAT                                                    #
# --------------------------------------------------------------------------- #
def import_data(csv_dir: str, postgres_uri: str) -> None:
    """Importe MovieLens 1M dans PostgreSQL."""

    engine = create_engine(postgres_uri)

    # 1) Créer/‐vérifier les tables d'abord
    create_tables(engine)

    # 2) ---------- USERS ----------
    logger.info("👥 Import des utilisateurs…")
    users_df = pd.read_csv(
        os.path.join(csv_dir, "users.dat"),
        sep="::",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        engine="python",
    )
    users_df.to_sql("users", engine, if_exists="append", index=False)
    logger.info(f"   → {len(users_df):,} utilisateurs")

    # 3) ---------- MOVIES ----------
    logger.info("🎬 Import des films…")
    movies_df = pd.read_csv(
        os.path.join(csv_dir, "movies.dat"),
        sep="::",
        names=["movie_id", "title", "genres"],
        engine="python",
        encoding="latin1",
    )
    movies_df.to_sql("movies", engine, if_exists="append", index=False)
    logger.info(f"   → {len(movies_df):,} films")

    # 4) ---------- RATINGS ----------
    logger.info("⭐ Import des ratings… (1 M lignes)")
    ratings_df = pd.read_csv(
        os.path.join(csv_dir, "ratings.dat"),
        sep="::",
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python",
    )
    ratings_df.to_sql(
        "ratings",
        engine,
        if_exists="append",
        index=False,
        chunksize=200_000,
        method="multi",
    )
    logger.info(f"   → {len(ratings_df):,} ratings")

    # 5) ---------- STATISTIQUES ----------
    density = len(ratings_df) / (len(users_df) * len(movies_df)) * 100
    logger.info("\n📊 Statistiques d'import :")
    logger.info(f"• Utilisateurs : {len(users_df):,}")
    logger.info(f"• Films :       {len(movies_df):,}")
    logger.info(f"• Ratings :     {len(ratings_df):,}")
    logger.info(f"• Densité matrice : {density:.4f} %")
    logger.info("🎉 Import terminé avec succès !")



# --------------------------------------------------------------------------- #
# 3.  CLI                                                                     #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Import MovieLens 1M dans PostgreSQL"
    )
    parser.add_argument(
        "--csv_dir", default="data/raw", help="Répertoire contenant *.dat"
    )
    parser.add_argument(
        "--postgres_uri",
        default="postgresql://postgres:postgres@localhost:5432/cinematch",
        help="URI de connexion PostgreSQL",
    )
    args = parser.parse_args()

    # Vérifie la présence des fichiers
    for fname in ("users.dat", "movies.dat", "ratings.dat"):
        path = os.path.join(args.csv_dir, fname)
        if not os.path.exists(path):
            logger.error(f"❌ Fichier manquant : {path}")
            return

    try:
        import_data(args.csv_dir, args.postgres_uri)
    except Exception as exc:
        logger.error(f"❌ Erreur lors de l’import : {exc}")


if __name__ == "__main__":
    main()
