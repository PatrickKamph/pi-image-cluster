FROM python:3.11-slim

# Install system utilities needed by C-extensions on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Pre-download the YOLO11 model weights so the container doesn't download it at runtime
RUN python3 -c "from ultralytics import YOLO; YOLO('models/yolo11s.pt')"

# Default command to run
CMD ["python3", "-m", "cluster.driver"]
