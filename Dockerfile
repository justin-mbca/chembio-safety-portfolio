# Multi-stage build for smaller final image
FROM python:3.11-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy minimal requirements and install in isolated environment
COPY requirements-railway.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# Final stage - smaller image
FROM python:3.11-slim

# Set environment variables for Railway
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/home/chemapp/.local/bin:$PATH"

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

# Create user first
RUN useradd -m -u 1000 chemapp

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/chemapp/.local

# Copy only essential application files
COPY simple_api.py .
COPY risk_classifier.py .
COPY run.py .
COPY src/ ./src/

# Create logs directory
RUN mkdir -p logs && chown -R chemapp:chemapp /app
USER chemapp

# Expose port (Railway will set PORT environment variable)
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-3000}/health || exit 1

# Run application (Railway provides PORT environment variable)
CMD ["python", "run.py"]
