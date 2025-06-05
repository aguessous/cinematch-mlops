FROM python:3.9-slim

RUN pip install mlflow psycopg2-binary

WORKDIR /mlflow

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0"]