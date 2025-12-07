#!/bin/bash

echo "========================================"
echo "PhishGuard Setup Script"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8 or 3.9"
    exit 1
fi

echo "[1/4] Python found:"
python3 --version
echo ""

echo "[2/4] Installing Model 1 dependencies..."
cd Model1_EmailContent
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install Model 1 dependencies"
    exit 1
fi
cd ..

echo ""
echo "[3/4] Installing Model 2 dependencies..."
cd Model2_URLDetection/Source_Code
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install Model 2 dependencies"
    exit 1
fi
cd ../..

echo ""
echo "[4/4] Installing API server dependencies..."
cd api_server
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install API server dependencies"
    exit 1
fi
cd ..

echo ""
echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Train the models (if not already trained):"
echo "   python3 train_both_models.py"
echo ""
echo "2. Start the API server:"
echo "   ./start_server.sh"
echo ""
echo "3. Load the Chrome extension in Chrome"
echo ""






