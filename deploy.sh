#!/bin/bash

# ChemBio SafeGuard Production Deployment Script
# This script provides easy deployment options for different environments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}"
    echo "🔬 ChemBio SafeGuard Production Deployment"
    echo "=========================================="
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_dependencies() {
    echo "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "All dependencies are installed"
}

deploy_docker() {
    echo "🐳 Deploying with Docker Compose..."
    
    # Create necessary directories
    mkdir -p logs ssl
    
    # Build and start services
    docker-compose -f docker-compose.prod.yml up --build -d
    
    echo "Waiting for services to start..."
    sleep 10
    
    # Check health
    if curl -f http://localhost:3000/health &> /dev/null; then
        print_success "API is healthy"
    else
        print_error "API health check failed"
        docker-compose -f docker-compose.prod.yml logs chembio-api
        exit 1
    fi
    
    if curl -f http://localhost:3001 &> /dev/null; then
        print_success "Frontend is healthy"
    else
        print_error "Frontend health check failed"
        docker-compose -f docker-compose.prod.yml logs chembio-frontend
        exit 1
    fi
    
    print_success "Docker deployment completed successfully!"
    echo ""
    echo "🌐 Access your application at:"
    echo "   Frontend: http://localhost:3001"
    echo "   API: http://localhost:3000"
    echo "   API Docs: http://localhost:3000/docs"
    echo ""
    echo "📋 Management commands:"
    echo "   View logs: docker-compose -f docker-compose.prod.yml logs -f"
    echo "   Stop: docker-compose -f docker-compose.prod.yml down"
    echo "   Restart: docker-compose -f docker-compose.prod.yml restart"
}

deploy_local() {
    echo "🖥️ Setting up local production deployment..."
    
    # Check Python version
    python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
    if (( $(echo "$python_version < 3.8" | bc -l) )); then
        print_error "Python 3.8+ is required. Current version: $python_version"
        exit 1
    fi
    
    # Create virtual environment
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    
    # Install dependencies
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Check Redis
    if ! pgrep redis-server > /dev/null; then
        echo "Starting Redis server..."
        redis-server --daemonize yes
    fi
    
    # Create production environment file
    cat > .env.production << EOF
ENV=production
DEBUG=false
API_HOST=0.0.0.0
API_PORT=3000
FRONTEND_PORT=3001
REDIS_URL=redis://localhost:6379
LOG_LEVEL=INFO
MAX_WORKERS=4
EOF
    
    # Create systemd service (optional)
    if command -v systemctl &> /dev/null; then
        echo "Creating systemd service..."
        sudo tee /etc/systemd/system/chembio-safety.service << EOF
[Unit]
Description=ChemBio Safety System
After=network.target

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/.venv/bin
EnvironmentFile=$(pwd)/.env.production
ExecStart=$(pwd)/.venv/bin/python run.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
        
        sudo systemctl daemon-reload
        sudo systemctl enable chembio-safety
        sudo systemctl start chembio-safety
        
        print_success "Systemd service created and started"
    else
        # Start manually
        source .env.production
        nohup python run.py > logs/api.log 2>&1 &
        API_PID=$!
        echo $API_PID > /tmp/chembio-api.pid
        
        cd frontend
        nohup python -m http.server 3001 > ../logs/frontend.log 2>&1 &
        FRONTEND_PID=$!
        echo $FRONTEND_PID > /tmp/chembio-frontend.pid
        
        print_success "Services started manually"
    fi
    
    print_success "Local production deployment completed!"
}

show_status() {
    echo "📊 System Status:"
    echo "=================="
    
    if docker-compose -f docker-compose.prod.yml ps | grep -q "Up"; then
        echo "🐳 Docker Deployment: Running"
        docker-compose -f docker-compose.prod.yml ps
    elif systemctl is-active --quiet chembio-safety 2>/dev/null; then
        echo "🖥️ Local Deployment: Running (systemd)"
        systemctl status chembio-safety --no-pager
    elif [ -f /tmp/chembio-api.pid ] && kill -0 $(cat /tmp/chembio-api.pid) 2>/dev/null; then
        echo "🖥️ Local Deployment: Running (manual)"
        echo "API PID: $(cat /tmp/chembio-api.pid)"
        echo "Frontend PID: $(cat /tmp/chembio-frontend.pid)"
    else
        echo "❌ No deployment detected"
    fi
    
    echo ""
    echo "🌐 Health Checks:"
    if curl -f http://localhost:3000/health &> /dev/null; then
        print_success "API: Healthy"
    else
        print_error "API: Not responding"
    fi
    
    if curl -f http://localhost:3001 &> /dev/null; then
        print_success "Frontend: Healthy"
    else
        print_error "Frontend: Not responding"
    fi
}

stop_services() {
    echo "🛑 Stopping ChemBio SafeGuard services..."
    
    # Stop Docker deployment
    if docker-compose -f docker-compose.prod.yml ps | grep -q "Up"; then
        docker-compose -f docker-compose.prod.yml down
        print_success "Docker services stopped"
    fi
    
    # Stop systemd service
    if systemctl is-active --quiet chembio-safety 2>/dev/null; then
        sudo systemctl stop chembio-safety
        print_success "Systemd service stopped"
    fi
    
    # Stop manual processes
    if [ -f /tmp/chembio-api.pid ]; then
        kill $(cat /tmp/chembio-api.pid) 2>/dev/null || true
        rm -f /tmp/chembio-api.pid
        print_success "API process stopped"
    fi
    
    if [ -f /tmp/chembio-frontend.pid ]; then
        kill $(cat /tmp/chembio-frontend.pid) 2>/dev/null || true
        rm -f /tmp/chembio-frontend.pid
        print_success "Frontend process stopped"
    fi
}

show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  docker     Deploy using Docker Compose (recommended)"
    echo "  local      Deploy locally with systemd or manual process"
    echo "  status     Show current deployment status"
    echo "  stop       Stop all services"
    echo "  help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 docker    # Deploy with Docker"
    echo "  $0 local     # Deploy locally"
    echo "  $0 status    # Check status"
    echo "  $0 stop      # Stop all services"
}

# Main execution
print_header

case "${1:-help}" in
    docker)
        check_dependencies
        deploy_docker
        ;;
    local)
        deploy_local
        ;;
    status)
        show_status
        ;;
    stop)
        stop_services
        ;;
    help|*)
        show_help
        ;;
esac
