FROM python:3.11-slim

WORKDIR /app

# Build this from the PROJECT ROOT, not the backend/ folder:
#   docker build -f backend/Dockerfile -t fraud-backend .

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY ml /ml
ENV ML_DIR=/ml

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
