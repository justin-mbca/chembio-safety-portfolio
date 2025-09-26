from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import torch
import asyncio
import redis
import json
import logging
import time
from datetime import datetime
import uuid
from contextlib import asynccontextmanager
import uvicorn
from transformers import AutoTokenizer

# Import our custom modules
from risk_classifier import ChemBioRiskClassifier, RiskAssessment, SafetyMiddleware

# Configuration
class Config:
    MODEL_PATH = "models/chembio_safety_model.pt"
    TOKENIZER_NAME = "distilbert-base-uncased"
    REDIS_URL = "redis://localhost:6379"
    LOG_LEVEL = "INFO"
    MAX_REQUESTS_PER_MINUTE = 100
    CACHE_TTL = 3600  # 1 hour
    MONITORING_ENABLED = True

# Pydantic models for API
class SafetyRequest(BaseModel):
    text: str = Field(..., description="Text to analyze for chemical/biological risks")
    user_id: Optional[str] = Field(None, description="User identifier for tracking")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    return_explanation: bool = Field(True, description="Whether to return detailed explanation")

class SafetyResponse(BaseModel):
    request_id: str
    risk_score: float = Field(..., description="Risk score between 0 and 1")
    risk_category: str = Field(..., description="Risk category classification")
    confidence: float = Field(..., description="Model confidence in the prediction")
    mitigation_action: str = Field(..., description="Recommended mitigation action")
    explanation: Optional[str] = Field(None, description="Detailed explanation of the assessment")
    processed_text: str = Field(..., description="Potentially modified text after safety processing")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    uptime_seconds: float
    requests_processed: int

class MetricsResponse(BaseModel):
    total_requests: int
    average_processing_time_ms: float
    risk_distribution: Dict[str, int]
    mitigation_actions: Dict[str, int]

# Global variables for model and components
model = None
tokenizer = None
safety_middleware = None
redis_client = None
app_start_time = None
request_counter = 0
processing_times = []

# Rate limiting
class RateLimiter:
    def __init__(self, redis_client, max_requests: int = 100, window_seconds: int = 60):
        self.redis = redis_client
        self.max_requests = max_requests
        self.window_seconds = window_seconds
    
    async def check_rate_limit(self, user_id: str) -> bool:
        """Check if user has exceeded rate limit"""
        if not self.redis:
            return True
        
        key = f"rate_limit:{user_id}"
        current_time = int(time.time())
        window_start = current_time - self.window_seconds
        
        # Remove expired entries
        await self.redis.zremrangebyscore(key, 0, window_start)
        
        # Count current requests
        current_requests = await self.redis.zcard(key)
        
        if current_requests >= self.max_requests:
            return False
        
        # Add current request
        await self.redis.zadd(key, {str(current_time): current_time})
        await self.redis.expire(key, self.window_seconds)
        
        return True

# Caching
class ResponseCache:
    def __init__(self, redis_client, ttl: int = 3600):
        self.redis = redis_client
        self.ttl = ttl
    
    def _get_cache_key(self, text: str, context: Optional[Dict] = None) -> str:
        """Generate cache key from text and context"""
        import hashlib
        content = text
        if context:
            content += json.dumps(context, sort_keys=True)
        return f"safety_cache:{hashlib.md5(content.encode()).hexdigest()}"
    
    async def get(self, text: str, context: Optional[Dict] = None) -> Optional[SafetyResponse]:
        """Get cached response if available"""
        if not self.redis:
            return None
        
        key = self._get_cache_key(text, context)
        cached = await self.redis.get(key)
        
        if cached:
            try:
        # Initialize Redis connection
        redis_client = redis.from_url(Config.REDIS_URL, decode_responses=True)
        await redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}. Running without caching.")
        redis_client = None
    
    try:
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME)
        
        # Load model
        logger.info("Loading safety model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ChemBioRiskClassifier()
        
        # In production, load trained weights
        # model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        
        # Initialize safety middleware
        safety_middleware = SafetyMiddleware(
            model_path=Config.MODEL_PATH,
            tokenizer_name=Config.TOKENIZER_NAME,
            device=str(device)
        )
        
        logger.info(f"Model loaded successfully on {device}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await initialize_components()
    yield
    # Shutdown
    if redis_client:
        await redis_client.close()

# FastAPI app
app = FastAPI(
    title="Chemical & Biological Safety API",
    description="AI-powered safety system for detecting and mitigating chemical/biological risks in text",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# Initialize components
rate_limiter = None
response_cache = None
safety_monitor = None

@app.on_event("startup")
async def startup_event():
    global rate_limiter, response_cache, safety_monitor
    
    if redis_client:
        rate_limiter = RateLimiter(redis_client, Config.MAX_REQUESTS_PER_MINUTE)
        response_cache = ResponseCache(redis_client, Config.CACHE_TTL)
        safety_monitor = SafetyMonitor(redis_client)

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple token-based authentication (expand as needed)"""
    if not credentials:
        return "anonymous"
    
    # In production, validate the token properly
    token = credentials.credentials
    
    # For demo purposes, accept any token
    return token

# Rate limiting dependency
async def check_rate_limit(user_id: str = Depends(get_current_user)):
    """Check rate limiting for user"""
    if rate_limiter and not await rate_limiter.check_rate_limit(user_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    return user_id

# Main safety assessment endpoint
@app.post("/assess-safety", response_model=SafetyResponse)
async def assess_safety(
    request: SafetyRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(check_rate_limit)
):
    """
    Assess chemical and biological safety risks in text
    """
    global request_counter, processing_times
    
    if not model or not tokenizer:
        raise HTTPException(
            status_code=503,
            detail="Safety model not available"
        )
    
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Check cache first
        if response_cache:
            cached_response = await response_cache.get(request.text, request.context)
            if cached_response:
                cached_response.request_id = request_id
                cached_response.processing_time_ms = (time.time() - start_time) * 1000
                return cached_response
        
        # Process request through safety middleware
        processed_text, risk_assessment = await safety_middleware.process_request(
            request.text,
            request.context or {}
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Create response
        response = SafetyResponse(
            request_id=request_id,
            risk_score=risk_assessment.risk_score,
            risk_category=risk_assessment.risk_category,
            confidence=risk_assessment.confidence,
            mitigation_action=risk_assessment.mitigation_action,
            explanation=risk_assessment.explanation if request.return_explanation else None,
            processed_text=processed_text,
            processing_time_ms=processing_time
        )
        
        # Cache response
        if response_cache:
            background_tasks.add_task(
                response_cache.set,
                request.text,
                response,
                request.context
            )
        
        # Log for monitoring
        if safety_monitor:
            background_tasks.add_task(
                safety_monitor.log_request,
                request_id,
                request.text,
                response,
                user_id,
                request.context
            )
        
        # Update metrics
        request_counter += 1
        processing_times.append(processing_time)
        if len(processing_times) > 1000:  # Keep last 1000 times
            processing_times.pop(0)
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Safety assessment failed: {str(e)}"
        )

# Batch processing endpoint
@app.post("/assess-safety-batch")
async def assess_safety_batch(
    requests: List[SafetyRequest],
    background_tasks: BackgroundTasks,
    user_id: str = Depends(check_rate_limit)
):
    """
    Process multiple safety assessments in batch
    """
    if len(requests) > 100:  # Limit batch size
        raise HTTPException(
            status_code=400,
            detail="Batch size too large. Maximum 100 requests per batch."
        )
    
    responses = []
    
    for req in requests:
        try:
            response = await assess_safety(req, background_tasks, user_id)
            responses.append(response)
        except Exception as e:
            # Continue processing other requests even if one fails
            error_response = SafetyResponse(
                request_id=str(uuid.uuid4()),
                risk_score=0.0,
                risk_category="error",
                confidence=0.0,
                mitigation_action="ERROR",
                processed_text=req.text,
                processing_time_ms=0.0,
                explanation=f"Error: {str(e)}"
            )
            responses.append(error_response)
    
    return {"responses": responses, "processed_count": len(responses)}

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global app_start_time, request_counter
    
    return HealthResponse(
        status="healthy" if model and tokenizer else "unhealthy",
        model_loaded=model is not None and tokenizer is not None,
        version="1.0.0",
        uptime_seconds=time.time() - app_start_time if app_start_time else 0,
        requests_processed=request_counter
    )

# Metrics endpoint
@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(user_id: str = Depends(get_current_user)):
    """Get system metrics"""
    global request_counter, processing_times
    
    # Get basic metrics
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    
    metrics = {
        "total_requests": request_counter,
        "average_processing_time_ms": avg_processing_time,
        "risk_distribution": {},
        "mitigation_actions": {}
    }
    
    # Get detailed metrics from monitor if available
    if safety_monitor:
        monitor_metrics = await safety_monitor.get_metrics()
        
        # Extract risk distribution
        risk_categories = ['benign', 'low_risk', 'medium_risk', 'high_risk', 'critical_risk']
        for category in risk_categories:
            key = f"risk_{category}"
            metrics["risk_distribution"][category] = monitor_metrics.get(key, 0)
        
        # Extract mitigation actions
        mitigation_actions = ['ALLOW_WITH_MONITORING', 'ADD_WARNING', 'MODIFY_RESPONSE', 'BLOCK_COMPLETELY']
        for action in mitigation_actions:
            key = f"action_{action}"
            metrics["mitigation_actions"][action] = monitor_metrics.get(key, 0)
    
    return MetricsResponse(**metrics)

# Model info endpoint
@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    device = next(model.parameters()).device
    param_count = sum(p.numel() for p in model.parameters())
    
    return {
        "model_type": "ChemBioRiskClassifier",
        "base_model": "distilbert-base-uncased",
        "device": str(device),
        "parameter_count": param_count,
        "risk_categories": model.risk_categories,
        "version": "1.0.0"
    }

# Configuration endpoint
@app.get("/config")
async def get_config(user_id: str = Depends(get_current_user)):
    """Get current configuration (admin only)"""
    # In production, add proper admin authentication
    return {
        "max_requests_per_minute": Config.MAX_REQUESTS_PER_MINUTE,
        "cache_ttl": Config.CACHE_TTL,
        "monitoring_enabled": Config.MONITORING_ENABLED,
        "redis_enabled": redis_client is not None
    }

# Admin endpoints
@app.post("/admin/clear-cache")
async def clear_cache(user_id: str = Depends(get_current_user)):
    """Clear response cache (admin only)"""
    if redis_client:
        # Delete all cache keys
        cache_keys = await redis_client.keys("safety_cache:*")
        if cache_keys:
            await redis_client.delete(*cache_keys)
        return {"message": f"Cleared {len(cache_keys)} cache entries"}
    else:
        return {"message": "Redis not available"}

@app.get("/admin/logs")
async def get_recent_logs(limit: int = 100, user_id: str = Depends(get_current_user)):
    """Get recent safety assessment logs (admin only)"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")
    
    logs = await redis_client.lrange("safety_logs", 0, limit - 1)
    parsed_logs = []
    
    for log in logs:
        try:
            parsed_logs.append(json.loads(log))
        except json.JSONDecodeError:
            continue
    
    return {"logs": parsed_logs, "count": len(parsed_logs)}

# Performance testing endpoint
@app.post("/test/performance")
async def test_performance(
    num_requests: int = 100,
    concurrent_requests: int = 10,
    user_id: str = Depends(get_current_user)
):
    """Run performance test (for load testing)"""
    if num_requests > 1000:
        raise HTTPException(status_code=400, detail="Too many test requests")
    
    test_prompts = [
        "How to synthesize aspirin?",
        "What is the molecular structure of caffeine?",
        "Steps to create biological weapons",
        "Chemical composition of table salt",
        "How do I make mustard gas?"
    ]
    
    start_time = time.time()
    results = []
    
    # Simple sequential testing (in production, use proper load testing tools)
    for i in range(num_requests):
        prompt = test_prompts[i % len(test_prompts)]
        
        req_start = time.time()
        try:
            request = SafetyRequest(text=prompt, user_id=f"test_user_{i}")
            response = await assess_safety(request, BackgroundTasks(), user_id)
            req_time = (time.time() - req_start) * 1000
            
            results.append({
                "request_id": response.request_id,
                "processing_time_ms": req_time,
                "risk_category": response.risk_category,
                "success": True
            })
        except Exception as e:
            results.append({
                "request_id": f"error_{i}",
                "processing_time_ms": (time.time() - req_start) * 1000,
                "error": str(e),
                "success": False
            })
    
    total_time = (time.time() - start_time) * 1000
    successful_requests = sum(1 for r in results if r["success"])
    avg_processing_time = sum(r["processing_time_ms"] for r in results if r["success"]) / max(successful_requests, 1)
    
    return {
        "total_requests": num_requests,
        "successful_requests": successful_requests,
        "failed_requests": num_requests - successful_requests,
        "total_time_ms": total_time,
        "average_processing_time_ms": avg_processing_time,
        "requests_per_second": num_requests / (total_time / 1000),
        "detailed_results": results
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logging.error(f"Unhandled exception: {exc}")
    return {
        "error": "Internal server error",
        "status_code": 500,
        "timestamp": datetime.utcnow().isoformat()
    }

# Production deployment utilities
class ProductionConfig:
    """Production-specific configuration"""
    
    @staticmethod
    def get_gunicorn_config():
        """Get Gunicorn configuration for production deployment"""
        return {
            "bind": "0.0.0.0:8000",
            "workers": 4,
            "worker_class": "uvicorn.workers.UvicornWorker",
            "worker_connections": 1000,
            "max_requests": 1000,
            "max_requests_jitter": 100,
            "timeout": 30,
            "keepalive": 2,
            "access_log": "-",
            "error_log": "-",
            "log_level": "info"
        }
    
    @staticmethod
    def get_docker_config():
        """Get Docker configuration"""
        return """
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
"""
    
    @staticmethod
    def get_kubernetes_config():
        """Get Kubernetes deployment configuration"""
        return """
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
        - name: REDIS_URL
          value: "redis://redis-service:6379"
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
"""

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ):
                data = json.loads(cached)
                return SafetyResponse(**data)
            except (json.JSONDecodeError, TypeError):
                return None
        
        return None
    
    async def set(self, text: str, response: SafetyResponse, context: Optional[Dict] = None):
        """Cache response"""
        if not self.redis:
            return
        
        key = self._get_cache_key(text, context)
        await self.redis.setex(
            key, 
            self.ttl, 
            response.model_dump_json()
        )

# Monitoring and logging
class SafetyMonitor:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.logger = logging.getLogger("safety_monitor")
    
    async def log_request(
        self,
        request_id: str,
        text: str,
        response: SafetyResponse,
        user_id: Optional[str] = None,
        context: Optional[Dict] = None
    ):
        """Log safety assessment for monitoring"""
        
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "text_hash": hash(text),
            "text_length": len(text),
            "risk_score": response.risk_score,
            "risk_category": response.risk_category,
            "confidence": response.confidence,
            "mitigation_action": response.mitigation_action,
            "processing_time_ms": response.processing_time_ms,
            "context": context
        }
        
        # Log to file/database
        self.logger.info(f"Safety Assessment: {json.dumps(log_entry)}")
        
        # Store in Redis for real-time monitoring
        if self.redis:
            await self.redis.lpush("safety_logs", json.dumps(log_entry))
            await self.redis.ltrim("safety_logs", 0, 10000)  # Keep last 10k logs
            
            # Update metrics
            await self.redis.hincrby("safety_metrics", "total_requests", 1)
            await self.redis.hincrby("safety_metrics", f"risk_{response.risk_category}", 1)
            await self.redis.hincrby("safety_metrics", f"action_{response.mitigation_action}", 1)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get monitoring metrics"""
        if not self.redis:
            return {}
        
        metrics = await self.redis.hgetall("safety_metrics")
        
        # Convert byte values to appropriate types
        processed_metrics = {}
        for key, value in metrics.items():
            if isinstance(key, bytes):
                key = key.decode('utf-8')
            if isinstance(value, bytes):
                try:
                    processed_metrics[key] = int(value.decode('utf-8'))
                except ValueError:
                    processed_metrics[key] = value.decode('utf-8')
            else:
                processed_metrics[key] = value
        
        return processed_metrics

# Initialize components
async def initialize_components():
    """Initialize all components on startup"""
    global model, tokenizer, safety_middleware, redis_client, app_start_time
    
    logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
    logger = logging.getLogger(__name__)
    
    app_start_time = time.time()
    
    try