# Production Deployment Guide for ChemBio SafeGuard

## 🚀 Deployment Options Overview

Choose the deployment option that best fits your infrastructure and requirements:

1. **🐳 Docker Deployment** - Containerized deployment (Recommended)
2. **☁️ Cloud Deployment** - AWS, GCP, Azure
3. **🔧 Traditional VPS** - Ubuntu/CentOS server deployment
4. **📦 Kubernetes** - Container orchestration for scale
5. **🖥️ Local Production** - Single machine production setup

---

## 🐳 Option 1: Docker Deployment (Recommended)

### Step 1: Create Production Dockerfile

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 chemapp && chown -R chemapp:chemapp /app
USER chemapp

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Run application
CMD ["python", "run.py"]
```

### Step 2: Create Docker Compose for Full Stack

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  chembio-api:
    build: .
    ports:
      - "3000:3000"
    environment:
      - ENV=production
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=INFO
    depends_on:
      - redis
    restart: unless-stopped
    volumes:
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  chembio-frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "3001:3001"
    depends_on:
      - chembio-api
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 5s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - chembio-api
      - chembio-frontend
    restart: unless-stopped

volumes:
  redis_data:
```

### Step 3: Deploy Commands

```bash
# Build and deploy
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f chembio-api

# Scale API servers
docker-compose -f docker-compose.prod.yml up -d --scale chembio-api=3
```

---

## ☁️ Option 2: AWS Cloud Deployment

### A. AWS ECS Fargate Deployment

```bash
# 1. Build and push to ECR
aws ecr create-repository --repository-name chembio-safety

# Get login token
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-west-2.amazonaws.com

# Build and tag
docker build -t chembio-safety .
docker tag chembio-safety:latest <account-id>.dkr.ecr.us-west-2.amazonaws.com/chembio-safety:latest

# Push to ECR
docker push <account-id>.dkr.ecr.us-west-2.amazonaws.com/chembio-safety:latest
```

### B. ECS Task Definition

```json
{
  "family": "chembio-safety-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "chembio-safety-api",
      "image": "<account-id>.dkr.ecr.us-west-2.amazonaws.com/chembio-safety:latest",
      "portMappings": [
        {
          "containerPort": 3000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENV",
          "value": "production"
        },
        {
          "name": "REDIS_URL",
          "value": "redis://chembio-redis.abc123.cache.amazonaws.com:6379"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/chembio-safety",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:3000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

### C. CloudFormation Template

```yaml
# cloudformation.yml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'ChemBio SafeGuard Production Deployment'

Resources:
  # VPC and Networking
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      EnableDnsSupport: true

  # ECS Cluster
  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: chembio-safety-cluster

  # Application Load Balancer
  LoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: chembio-safety-alb
      Scheme: internet-facing
      Type: application
      Subnets: [!Ref PublicSubnet1, !Ref PublicSubnet2]

  # ECS Service
  ECSService:
    Type: AWS::ECS::Service
    Properties:
      ServiceName: chembio-safety-service
      Cluster: !Ref ECSCluster
      TaskDefinition: !Ref TaskDefinition
      DesiredCount: 2
      LaunchType: FARGATE
      NetworkConfiguration:
        AwsvpcConfiguration:
          SecurityGroups: [!Ref SecurityGroup]
          Subnets: [!Ref PrivateSubnet1, !Ref PrivateSubnet2]

  # Auto Scaling
  ServiceScalingTarget:
    Type: AWS::ApplicationAutoScaling::ScalableTarget
    Properties:
      MaxCapacity: 10
      MinCapacity: 2
      ResourceId: !Sub service/${ECSCluster}/${ECSService.Name}
      RoleARN: !Sub arn:aws:iam::${AWS::AccountId}:role/application-autoscaling-ecs-service
      ScalableDimension: ecs:service:DesiredCount
      ServiceNamespace: ecs
```

---

## 🔧 Option 3: Traditional VPS Deployment

### Ubuntu/CentOS Server Setup

```bash
# 1. Initial server setup
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3-pip nginx redis-server curl git

# 2. Create application user
sudo useradd -m -s /bin/bash chemapp
sudo mkdir /opt/chembio-safety
sudo chown chemapp:chemapp /opt/chembio-safety

# 3. Deploy application
sudo -u chemapp git clone https://github.com/justin-mbca/chembio-safety-portfolio.git /opt/chembio-safety
cd /opt/chembio-safety

# 4. Setup Python environment
sudo -u chemapp python3.11 -m venv venv
sudo -u chemapp ./venv/bin/pip install -r requirements.txt

# 5. Create systemd service
sudo tee /etc/systemd/system/chembio-api.service << EOF
[Unit]
Description=ChemBio Safety API
After=network.target redis.service

[Service]
Type=simple
User=chemapp
Group=chemapp
WorkingDirectory=/opt/chembio-safety
Environment=PATH=/opt/chembio-safety/venv/bin
ExecStart=/opt/chembio-safety/venv/bin/python run.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# 6. Start services
sudo systemctl daemon-reload
sudo systemctl enable chembio-api
sudo systemctl start chembio-api
sudo systemctl status chembio-api
```

### Nginx Configuration

```nginx
# /etc/nginx/sites-available/chembio-safety
server {
    listen 80;
    server_name your-domain.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    # API Backend
    location /api/ {
        proxy_pass http://127.0.0.1:3000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Frontend
    location / {
        proxy_pass http://127.0.0.1:3001/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Health check
    location /health {
        access_log off;
        proxy_pass http://127.0.0.1:3000/health;
    }
}
```

---

## 📦 Option 4: Kubernetes Deployment

### Kubernetes Manifests

```yaml
# k8s-namespace.yml
apiVersion: v1
kind: Namespace
metadata:
  name: chembio-safety

---
# k8s-configmap.yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: chembio-config
  namespace: chembio-safety
data:
  LOG_LEVEL: "INFO"
  ENV: "production"

---
# k8s-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chembio-api
  namespace: chembio-safety
spec:
  replicas: 3
  selector:
    matchLabels:
      app: chembio-api
  template:
    metadata:
      labels:
        app: chembio-api
    spec:
      containers:
      - name: api
        image: your-registry/chembio-safety:latest
        ports:
        - containerPort: 3000
        envFrom:
        - configMapRef:
            name: chembio-config
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10

---
# k8s-service.yml
apiVersion: v1
kind: Service
metadata:
  name: chembio-api-service
  namespace: chembio-safety
spec:
  selector:
    app: chembio-api
  ports:
  - port: 80
    targetPort: 3000
  type: LoadBalancer

---
# k8s-hpa.yml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: chembio-api-hpa
  namespace: chembio-safety
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: chembio-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Deploy to Kubernetes

```bash
# Apply manifests
kubectl apply -f k8s-namespace.yml
kubectl apply -f k8s-configmap.yml
kubectl apply -f k8s-deployment.yml
kubectl apply -f k8s-service.yml
kubectl apply -f k8s-hpa.yml

# Check deployment
kubectl get pods -n chembio-safety
kubectl get services -n chembio-safety

# Get external IP
kubectl get service chembio-api-service -n chembio-safety
```

---

## 🏭 Production Configuration

### Environment Variables

```bash
# .env.production
ENV=production
DEBUG=false

# API Configuration
API_HOST=0.0.0.0
API_PORT=3000
FRONTEND_PORT=3001

# Database
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your-super-secret-key-here
ALLOWED_HOSTS=your-domain.com,api.your-domain.com
CORS_ORIGINS=https://your-domain.com

# Logging
LOG_LEVEL=INFO
LOG_FILE=/app/logs/chembio-safety.log

# Performance
MAX_WORKERS=4
WORKER_TIMEOUT=30
MAX_REQUESTS=1000
```

### Production Startup Script

```bash
#!/bin/bash
# production-start.sh

echo "🚀 Starting ChemBio SafeGuard Production System..."

# Load environment variables
source .env.production

# Check dependencies
echo "Checking system requirements..."
python -c "import torch, transformers, fastapi; print('✅ Dependencies OK')"

# Start Redis if not running
redis-cli ping > /dev/null 2>&1 || {
    echo "Starting Redis..."
    redis-server --daemonize yes
}

# Start API with production settings
echo "Starting API server with $MAX_WORKERS workers..."
gunicorn run:app \
    --bind $API_HOST:$API_PORT \
    --workers $MAX_WORKERS \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout $WORKER_TIMEOUT \
    --max-requests $MAX_REQUESTS \
    --access-logfile logs/access.log \
    --error-logfile logs/error.log \
    --pid /tmp/chembio-api.pid \
    --daemon

# Start frontend server
echo "Starting frontend server..."
cd frontend
python -m http.server $FRONTEND_PORT > ../logs/frontend.log 2>&1 &
echo $! > /tmp/chembio-frontend.pid

echo "✅ Production system started successfully!"
echo "   API: http://$API_HOST:$API_PORT"
echo "   Frontend: http://$API_HOST:$FRONTEND_PORT"
echo "   Health: http://$API_HOST:$API_PORT/health"
```

---

## 📊 Monitoring & Maintenance

### Health Monitoring Script

```python
# monitoring/health_check.py
import requests
import time
import logging
from datetime import datetime

def check_health():
    """Comprehensive health check"""
    checks = {
        'api_health': check_api_health(),
        'redis_connection': check_redis(),
        'model_loading': check_model(),
        'response_time': check_response_time()
    }
    
    all_healthy = all(checks.values())
    
    if not all_healthy:
        send_alert(checks)
    
    return checks

def check_api_health():
    try:
        response = requests.get('http://localhost:3000/health', timeout=10)
        return response.status_code == 200
    except:
        return False

def check_response_time():
    try:
        start = time.time()
        requests.post('http://localhost:3000/assess-safety', 
                     json={"text": "test query"}, timeout=5)
        return (time.time() - start) < 2.0  # Should respond in < 2s
    except:
        return False

if __name__ == "__main__":
    result = check_health()
    print(f"Health check at {datetime.now()}: {result}")
```

### Backup Script

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/chembio-safety/$DATE"

mkdir -p $BACKUP_DIR

# Backup application
tar -czf "$BACKUP_DIR/app.tar.gz" /opt/chembio-safety \
    --exclude="*.pyc" --exclude="__pycache__" --exclude=".git"

# Backup logs
cp -r logs/ "$BACKUP_DIR/"

# Backup Redis data
redis-cli BGSAVE
cp /var/lib/redis/dump.rdb "$BACKUP_DIR/"

echo "✅ Backup completed: $BACKUP_DIR"

# Cleanup old backups (keep last 7 days)
find /backup/chembio-safety -type d -mtime +7 -exec rm -rf {} \;
```

---

## 🔒 Security Considerations

1. **HTTPS/TLS**: Always use SSL certificates in production
2. **Firewall**: Restrict access to necessary ports only
3. **Authentication**: Implement API keys or OAuth for production use
4. **Rate Limiting**: Configure rate limits to prevent abuse
5. **Security Headers**: Add security headers via nginx/load balancer
6. **Regular Updates**: Keep dependencies and system packages updated

---

**Choose your deployment option and follow the corresponding guide. For most use cases, Docker deployment (Option 1) is recommended for its simplicity and portability.**
