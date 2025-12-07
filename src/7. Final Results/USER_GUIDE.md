# PhishGuard - User Guide

## 🚀 Quick Start (For End Users)

### Step 1: Setup (One-Time)

**Windows:**
1. Double-click `setup.bat`
2. Wait for installation to complete
3. Close the window when done

**Mac/Linux:**
1. Open terminal in this folder
2. Run: `chmod +x setup.sh start_server.sh`
3. Run: `./setup.sh`
4. Wait for installation to complete

### Step 2: Train Models (One-Time, ~2-3 hours)

**Note:** If models are already trained, skip this step.

1. Open terminal/command prompt in this folder
2. Run: `python train_both_models.py` (or `python3` on Mac/Linux)
3. Wait for training to complete (~2-3 hours)
4. Models will be saved automatically

**Requirements:** NVIDIA GPU required for training

### Step 3: Start API Server

**Windows:**
- Double-click `start_server.bat`
- Keep the window open (minimize it)

**Mac/Linux:**
- Run: `./start_server.sh`
- Keep the terminal open

You should see: "Server starting on http://localhost:5000"

### Step 4: Load Chrome Extension

1. Open Chrome browser
2. Go to `chrome://extensions/`
3. Enable "Developer mode" (toggle in top-right)
4. Click "Load unpacked"
5. Select the `chrome_extension` folder
6. Extension is now installed!

### Step 5: Use It!

1. Go to Gmail or Outlook
2. Open any email
3. Extension will automatically analyze it
4. If phishing detected, you'll see a warning

---

## 📋 Requirements

- **Python 3.8 or 3.9** installed
- **Chrome browser**
- **NVIDIA GPU** (for training models - one-time only)
- **Internet connection** (for initial setup only)

---

## ❓ Troubleshooting

### "Python not found"
- Install Python from https://www.python.org/downloads/
- Make sure to check "Add Python to PATH" during installation

### "Models not found"
- Run `python train_both_models.py` to train models first
- Make sure training completed successfully

### "Port 5000 already in use"
- Close other applications using port 5000
- Or change port in `api_server/app.py` (line with `port=5000`)

### "Server failed to start"
- Make sure you ran `setup.bat` (or `setup.sh`) first
- Check that models are trained in `trained_models/` folder
- Try running `python api_server/app.py` directly to see error messages

### Extension not working
- Make sure API server is running (check the server window)
- Check extension is loaded in Chrome (`chrome://extensions/`)
- Open browser console (F12) to see any errors

---

## 🔄 Daily Use

**Every time you want to use PhishGuard:**

1. Start API server: Double-click `start_server.bat` (Windows) or run `./start_server.sh` (Mac/Linux)
2. Keep server running (minimize the window)
3. Use Chrome as normal - extension works automatically!

**To stop:**
- Close the server window (or press Ctrl+C in terminal)

---

## 📞 Need Help?

1. Check this guide first
2. See `SETUP_INSTRUCTIONS.md` for detailed setup
3. Check `TROUBLESHOOTING.md` for common issues

---

**That's it! Enjoy phishing protection!** 🛡️






