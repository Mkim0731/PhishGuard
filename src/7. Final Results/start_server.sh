#!/bin/bash

echo "========================================"
echo "PhishGuard API Server"
echo "========================================"
echo ""
echo "Starting server on http://localhost:5000"
echo ""
echo "Keep this terminal open while using the extension"
echo "Press Ctrl+C to stop the server"
echo ""
echo "========================================"
echo ""

cd "$(dirname "$0")"
cd api_server
python3 app.py

if [ $? -ne 0 ]; then
    echo ""
    echo "========================================"
    echo "ERROR: Server failed to start"
    echo "========================================"
    echo ""
    echo "Possible issues:"
    echo "1. Models not trained - Run: python3 train_both_models.py"
    echo "2. Dependencies not installed - Run: ./setup.sh"
    echo "3. Port 5000 already in use"
    echo ""
    read -p "Press Enter to exit..."
fi






