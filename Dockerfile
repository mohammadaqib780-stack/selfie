FROM python:3.10-slim

# Install system dependencies needed by deepface / opencv
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgtk2.0-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependencies
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY main.py .

# Expose API port
EXPOSE 8000

# Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
