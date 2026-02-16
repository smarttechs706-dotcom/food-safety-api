FROM python:3.10

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY autoencoder.pth .
COPY best.pt .
COPY best_image_classifier_98pct.pkl .
COPY best_text_classifier_87pct.pkl .
COPY image_classifier_91pct.pkl .
COPY text_classifier_v2_80pct.pkl .

# Copy application files
COPY inference.py .
COPY app.py .

# Create upload directory
RUN mkdir -p uploaded_images

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD uvicorn app:app --host 0.0.0.0 --port $PORT
