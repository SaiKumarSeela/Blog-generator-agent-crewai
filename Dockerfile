FROM python:3.12-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8085

# Default command to run FastAPI app
CMD ["uvicorn", "fastapi_main:app", "--host", "0.0.0.0", "--port", "8085"]
