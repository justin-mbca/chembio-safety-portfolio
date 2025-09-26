#!/bin/bash

# ChemBio SafeGuard System Launcher
# Starts both API server and frontend server

echo "🔬 Starting ChemBio SafeGuard System..."

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Warning: No virtual environment found. Please run 'python -m venv .venv' first."
fi

# Install requirements if needed
if [ -f "requirements.txt" ]; then
    echo "Installing/updating requirements..."
    pip install -r requirements.txt
fi

echo "Starting API server on port 3000..."
python run.py &
API_PID=$!

sleep 3

echo "Starting frontend server on port 3001..."
python src/api/frontend_server.py &
FRONTEND_PID=$!

echo ""
echo "🚀 ChemBio SafeGuard System is running:"
echo "   API Server: http://localhost:3000"
echo "   Frontend: http://localhost:3001"
echo ""
echo "Press Ctrl+C to stop all services..."

# Wait for interrupt signal
trap 'echo "Stopping services..."; kill $API_PID $FRONTEND_PID; exit' INT
wait