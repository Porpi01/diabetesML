FROM python:3.10.18-slim

RUN mkdir app
WORKDIR /app

COPY app.py ./
COPY requirements.txt ./
COPY modelos_pkl ./modelos_pkl

COPY templates ./templates

RUN pip install -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
