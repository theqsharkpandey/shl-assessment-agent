FROM python:3.11-slim

# Set environment variables for non-interactive behavior
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create a non-root user for security
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Install build dependencies required for FAISS and compiling native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Adjust permissions for data folders
RUN mkdir -p data/index && chown -R appuser:appuser /app/data

# Switch to non-root user
USER appuser

# Expose FastAPI default port
EXPOSE 8000

# Run the API via uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
