========================================
PhishGuard - Phishing Detection System
========================================

QUICK START GUIDE
==================

1. SETUP (One-time):
   - Windows: Double-click setup.bat
   - Mac/Linux: Run ./setup.sh in terminal

2. TRAIN MODELS (One-time, ~2-3 hours):
   - Run: python train_both_models.py
   - Requires: NVIDIA GPU
   - Note: If models already trained, skip this step

3. START SERVER:
   - Windows: Double-click start_server.bat
   - Mac/Linux: Run ./start_server.sh
   - Keep window open while using extension

4. LOAD CHROME EXTENSION:
   - Open Chrome
   - Go to chrome://extensions/
   - Enable "Developer mode"
   - Click "Load unpacked"
   - Select "chrome_extension" folder

5. USE IT:
   - Go to Gmail/Outlook
   - Extension works automatically!


REQUIREMENTS
============

- Python 3.8 or 3.9
- Chrome browser
- NVIDIA GPU (for training only)
- 20GB free disk space


FILES IN THIS PACKAGE
=====================

- setup.bat / setup.sh          Setup script
- start_server.bat / start_server.sh    Start API server
- train_both_models.py          Train both models
- trained_models/                Trained models (after training)
- api_server/                    API server code
- chrome_extension/              Chrome extension
- Model1_EmailContent/           Email content model
- Model2_URLDetection/           URL detection model


FOR DETAILED INSTRUCTIONS
=========================

See USER_GUIDE.md for complete instructions
See SETUP_INSTRUCTIONS.md for technical setup


TROUBLESHOOTING
===============

- Python not found: Install Python and add to PATH
- Models not found: Train models first (step 2)
- Server won't start: Check models are trained
- Extension not working: Make sure server is running


SUPPORT
=======

For issues, check:
1. USER_GUIDE.md
2. SETUP_INSTRUCTIONS.md
3. Troubleshooting sections


========================================
Enjoy phishing protection!
========================================






