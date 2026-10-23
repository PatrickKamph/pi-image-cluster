FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python3 -c "from ultralytics import YOLO; YOLO('models/yolo11s.pt')"

# Default command to run
CMD ["python3", "-m", "cluster.driver"]
