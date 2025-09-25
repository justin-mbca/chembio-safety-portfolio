# Deployment Guide

This guide provides comprehensive instructions for deploying the Chemical & Biological Safety System in various environments.

## Table of Contents
- [Quick Start](#quick-start)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Production Considerations](#production-considerations)
- [Monitoring & Maintenance](#monitoring--maintenance)

## Quick Start

### 1. Basic Setup
```bash
# Clone and setup
git clone https://github.com/justin-mbca/chembio-safety-portfolio.git
cd chembio-safety-portfolio
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run demonstration
python demo_script.py

# Start API server
python simple_api.py
```

### 2. Verify Installation
```bash
# Test API endpoints
python test_api.py

# Check health endpoint
curl http://localhost:8000/health
```

## Local Development

### Development Server
```bash
# Start with auto-reload
uvicorn simple_api:app --reload --port 8000

# Or use the built-in server
python simple_api.py
```

### Environment Variables
Create a `.env` file:
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
MODEL_PATH=models/chembio_safety_model.pt
LOG_LEVEL=INFO

# Redis (optional)
REDIS_URL=redis://localhost:6379

# Security
MAX_REQUESTS_PER_MINUTE=1000
```

## Docker Deployment

### 1. Build Docker Image
```bash
# Create Dockerfile
cat > Dockerfile << EOF
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "simple_api.py"]
EOF

# Build image
docker build -t chembio-safety-api .
```

### 2. Run Container
```bash
# Run single container
docker run -d \
  --name chembio-safety \
  -p 8000:8000 \
  -e LOG_LEVEL=INFO \
  chembio-safety-api

# Check container health
docker ps
docker logs chembio-safety
```

### 3. Docker Compose (with Redis)
```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=INFO
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

```bash
# Deploy with compose
docker-compose up -d

# Scale the API
docker-compose up -d --scale api=3
```

## Cloud Deployment

### AWS ECS Deployment
```json
{
  "family": "chembio-safety-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "chembio-safety-api",
      "image": "your-account.dkr.ecr.region.amazonaws.com/chembio-safety:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "LOG_LEVEL",
          "value": "INFO"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/chembio-safety",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Kubernetes Deployment
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chembio-safety-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: chembio-safety-api
  template:
    metadata:
      labels:
        app: chembio-safety-api
    spec:
      containers:
      - name: api
        image: chembio-safety-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10

---
apiVersion: v1
kind: Service
metadata:
  name: chembio-safety-service
spec:
  selector:
    app: chembio-safety-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

```bash
# Deploy to Kubernetes
kubectl apply -f k8s-deployment.yaml

# Scale deployment
kubectl scale deployment chembio-safety-api --replicas=5

# Check status
kubectl get pods -l app=chembio-safety-api
```

## Production Considerations

### Performance Optimization
```python
# gunicorn_config.py
bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
timeout = 30
keepalive = 2
preload_app = True
```

### Security Configuration
```bash
# Use environment variables for sensitive data
export MODEL_PATH="/secure/path/to/model"
export SECRET_KEY="your-secret-key"
export ALLOWED_HOSTS="api.yourcompany.com"

# Enable HTTPS
export SSL_CERT_PATH="/path/to/cert.pem"
export SSL_KEY_PATH="/path/to/key.pem"
```

### Load Balancing (nginx)
```nginx
upstream chembio_api {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

server {
    listen 80;
    server_name api.yourcompany.com;

    location / {
        proxy_pass http://chembio_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /health {
        access_log off;
        proxy_pass http://chembio_api/health;
    }
}
```

## Monitoring & Maintenance

### Health Monitoring
```bash
# Health check script
#!/bin/bash
HEALTH_URL="http://localhost:8000/health"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $HEALTH_URL)

if [ $RESPONSE -eq 200 ]; then
    echo "✅ API is healthy"
    exit 0
else
    echo "❌ API health check failed (HTTP $RESPONSE)"
    exit 1
fi
```

### Log Monitoring
```bash
# Centralized logging with ELK stack
# logstash.conf
input {
  file {
    path => "/var/log/chembio-safety/*.log"
    start_position => "beginning"
  }
}

filter {
  json {
    source => "message"
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "chembio-safety-%{+YYYY.MM.dd}"
  }
}
```

### Metrics Collection
```python
# metrics.py - Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active connections')

# Safety metrics
RISK_ASSESSMENTS = Counter('risk_assessments_total', 'Total risk assessments', ['risk_level'])
MITIGATION_ACTIONS = Counter('mitigation_actions_total', 'Mitigation actions taken', ['action_type'])
```

### Backup & Recovery
```bash
# Backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/chembio-safety/$DATE"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup model files
cp -r models/ $BACKUP_DIR/

# Backup configuration
cp .env $BACKUP_DIR/

# Backup logs
cp -r logs/ $BACKUP_DIR/

# Compress backup
tar -czf "${BACKUP_DIR}.tar.gz" -C /backup/chembio-safety $DATE

echo "✅ Backup completed: ${BACKUP_DIR}.tar.gz"
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```bash
   # Check model file permissions
   ls -la models/
   
   # Verify model path
   python -c "import os; print(os.path.exists('models/chembio_safety_model.pt'))"
   ```

2. **Memory Issues**
   ```bash
   # Monitor memory usage
   docker stats chembio-safety
   
   # Increase memory limits
   docker run -m 4g chembio-safety-api
   ```

3. **Performance Issues**
   ```bash
   # Check API performance
   curl -w "@curl-format.txt" -s http://localhost:8000/health
   
   # Monitor with htop
   htop -p $(pgrep -f uvicorn)
   ```

### Support

For deployment issues:
1. Check the [GitHub Issues](https://github.com/justin-mbca/chembio-safety-portfolio/issues)
2. Review the logs: `docker logs chembio-safety`
3. Verify environment configuration
4. Contact the maintainers for assistance

---

**Note:** This deployment guide assumes you have the necessary infrastructure permissions and security clearances for deploying AI safety systems in your organization.
