FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for OpenCV + YOLO)
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL model files
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

# Expose port
EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
