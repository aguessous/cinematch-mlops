FROM python:3.9-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential libgomp1 && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV OPENBLAS_NUM_THREADS=1
CMD ["python", "ml/train.py"]
