#!/bin/bash

# Script de téléchargement des données MovieLens 1M

DATA_DIR="data/raw"
MOVIELENS_URL="http://files.grouplens.org/datasets/movielens/ml-1m.zip"

echo "🎬 Téléchargement des données MovieLens 1M..."

# Création du répertoire
mkdir -p $DATA_DIR

# Téléchargement
cd $DATA_DIR
wget $MOVIELENS_URL

# Extraction
unzip ml-1m.zip

# Déplacement des fichiers
mv ml-1m/* .
rmdir ml-1m
rm ml-1m.zip

echo "✅ Données MovieLens téléchargées dans $DATA_DIR"
echo "📁 Fichiers disponibles:"
ls -la

echo "🐳 Vous pouvez maintenant lancer: docker compose up -d"