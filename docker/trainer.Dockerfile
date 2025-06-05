FROM python:3.9-slim

# ❶ dépendances système : git pour MLflow, build‑essential pour SciPy/implicit
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir 'numpy<2'
RUN pip install --no-cache-dir -r requirements.txt

# ❷ on copie le reste une seule fois
COPY . .

# ❸ bonne pratique : variable d’environnement pour OpenBLAS (déjà utilisée dans ton script)


ENV OPENBLAS_NUM_THREADS=1 IMPLICIT_NO_CHECK_BLAS=1

CMD ["python", "ml/train.py"]
