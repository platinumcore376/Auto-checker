FROM python:3.12-slim

# Install OpenCV and PDF processing dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
 && rm -rf /var/lib/apt/lists/*

# Create non-root user (DOCKER-02)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Ensure cache directories (huggingface, torch, matplotlib) and app directory are writable by appuser
RUN mkdir -p /home/appuser/.cache && chown -R appuser:appuser /home/appuser /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
