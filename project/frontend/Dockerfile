FROM python:3.11-slim

WORKDIR /app

# Build this from the PROJECT ROOT:
#   docker build -f frontend/Dockerfile -t fraud-frontend .

COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/dashboard.py .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
