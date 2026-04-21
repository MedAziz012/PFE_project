# 1. Use a slim Python image
FROM python:3.10-slim

# 2. Install OCR and PDF system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# 3. Set the work directory
WORKDIR /app

# 4. Copy requirements from the root
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy everything else
COPY . .

# 6. Start the app
# Note: If your FastAPI app is inside ai_service/main.py, 
# change the path to: ai_service.main:app
CMD ["uvicorn", "ai_service.main:app", "--host", "0.0.0.0", "--port", "8080"]
