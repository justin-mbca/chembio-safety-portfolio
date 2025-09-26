#!/bin/bash

# ChemBio SafeGuard - Complete System Startup Script
# This script starts both the API backend and web frontend

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
API_PORT=8000
FRONTEND_PORT=3001
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🚀 ChemBio SafeGuard System Startup${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to cleanup background processes
cleanup() {
    echo -e "\n${YELLOW}🛑 Shutting down services...${NC}"
    
    # Kill background processes if they exist
    if [[ -n "$API_PID" ]]; then
        kill $API_PID 2>/dev/null || true
        echo -e "${GREEN}✅ API server stopped${NC}"
    fi
    
    if [[ -n "$FRONTEND_PID" ]]; then
        kill $FRONTEND_PID 2>/dev/null || true
        echo -e "${GREEN}✅ Frontend server stopped${NC}"
    fi
    
    echo -e "${GREEN}👋 ChemBio SafeGuard system shutdown complete${NC}"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check Python environment
echo -e "${BLUE}🐍 Checking Python environment...${NC}"
cd "$PROJECT_DIR"

if [[ ! -d ".venv" ]]; then
    echo -e "${RED}❌ Virtual environment not found${NC}"
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv .venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi

# Activate virtual environment
source .venv/bin/activate
echo -e "${GREEN}✅ Virtual environment activated${NC}"

# Install/update dependencies
echo -e "${BLUE}📦 Installing dependencies...${NC}"
pip install -q -r requirements.txt
echo -e "${GREEN}✅ Dependencies installed${NC}"

# Check if ports are available
if check_port $API_PORT; then
    echo -e "${RED}❌ Port $API_PORT is already in use${NC}"
    echo -e "${YELLOW}💡 Please stop the existing service or use a different port${NC}"
    exit 1
fi

if check_port $FRONTEND_PORT; then
    echo -e "${RED}❌ Port $FRONTEND_PORT is already in use${NC}"
    echo -e "${YELLOW}💡 Please stop the existing service or use a different port${NC}"
    exit 1
fi

# Start API server
echo -e "\n${BLUE}🔧 Starting API server...${NC}"
python simple_api.py &
API_PID=$!

# Wait for API to be ready
echo -e "${YELLOW}⏳ Waiting for API server to start...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:$API_PORT/health >/dev/null 2>&1; then
        echo -e "${GREEN}✅ API server ready on http://localhost:$API_PORT${NC}"
        break
    fi
    
    if ! kill -0 $API_PID 2>/dev/null; then
        echo -e "${RED}❌ API server failed to start${NC}"
        exit 1
    fi
    
    sleep 1
    
    if [[ $i -eq 30 ]]; then
        echo -e "${RED}❌ API server timeout after 30 seconds${NC}"
        kill $API_PID 2>/dev/null || true
        exit 1
    fi
done

# Start frontend server
echo -e "\n${BLUE}🌐 Starting frontend server...${NC}"
python frontend_server.py --port $FRONTEND_PORT &
FRONTEND_PID=$!

# Wait for frontend to be ready
sleep 2
if kill -0 $FRONTEND_PID 2>/dev/null; then
    echo -e "${GREEN}✅ Frontend server ready on http://localhost:$FRONTEND_PORT${NC}"
else
    echo -e "${RED}❌ Frontend server failed to start${NC}"
    kill $API_PID 2>/dev/null || true
    exit 1
fi

# System ready
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}🎉 ChemBio SafeGuard System Ready!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${BLUE}📊 API Documentation: ${NC}http://localhost:$API_PORT/docs"
echo -e "${BLUE}🌐 Web Interface: ${NC}http://localhost:$FRONTEND_PORT"
echo -e "${BLUE}❤️  Health Check: ${NC}http://localhost:$API_PORT/health"
echo -e "${GREEN}========================================${NC}"
echo -e "${YELLOW}💡 Press Ctrl+C to stop all services${NC}"
echo -e ""

# Keep script running and monitor services
while true; do
    # Check if API is still running
    if ! kill -0 $API_PID 2>/dev/null; then
        echo -e "${RED}❌ API server stopped unexpectedly${NC}"
        break
    fi
    
    # Check if frontend is still running
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo -e "${RED}❌ Frontend server stopped unexpectedly${NC}"
        break
    fi
    
    sleep 5
done

cleanup
