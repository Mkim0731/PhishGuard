# Flash Drive Setup - Complete Portable Package

This guide helps you create a **fully portable** PhishGuard that works on any computer from a flash drive.

## 📦 What You'll Have

After setup, your flash drive will contain:
- ✅ Python (bundled, no installation needed)
- ✅ All dependencies (PyTorch, TensorFlow, etc.)
- ✅ Trained models (if already trained)
- ✅ API server (ready to run)
- ✅ Chrome extension (ready to load)

**Total size: ~15-20 GB** (but works on any computer!)

## 🚀 Setup Steps

### Step 1: Install All Dependencies

1. **Run the installation script:**
   ```
   Double-click: INSTALL_ALL_DEPENDENCIES.bat
   ```

2. **Wait for installation** (20-40 minutes):
   - PyTorch: ~10-15 minutes
   - TensorFlow: ~5-10 minutes
   - Other packages: ~5-10 minutes

3. **Verify installation:**
   ```bash
   python\python.exe -c "import torch; import tensorflow; print('OK: All installed!')"
   ```

### Step 2: Ensure Models Are Trained

If models are already trained, make sure `trained_models\` folder contains:
- `Model1\` - DistilBERT model files
- `Model2\` - CNN model files (model.h5, tokenizer.pkl)

If not trained, train them first (requires GPU):
```bash
python\python.exe train_both_models.py
```

### Step 3: Copy to Flash Drive

1. Copy **entire** `PhishGuard-main` folder to flash drive
2. Keep folder structure intact
3. Make sure `python\` folder is included

### Step 4: Test on Flash Drive

1. Plug flash drive into any computer
2. Navigate to: `flash_drive:\PhishGuard-main\src\Final_Project\`
3. Run: `api_server\start_server.bat`
4. Server should start on `http://localhost:5000`

## 💻 Using on Another Computer

### For Your Friend (No Setup Needed!)

1. **Plug in flash drive**
2. **Open Chrome browser**
3. **Load extension:**
   - Go to `chrome://extensions/`
   - Enable "Developer mode"
   - Click "Load unpacked"
   - Select: `flash_drive:\PhishGuard-main\src\Final_Project\chrome_extension\`

4. **Start API server:**
   - Navigate to: `flash_drive:\PhishGuard-main\src\Final_Project\`
   - Double-click: `api_server\start_server.bat`
   - Keep window open

5. **Use extension:**
   - Go to Gmail/Outlook
   - Extension will analyze emails automatically!

## 📁 Folder Structure (After Setup)

```
PhishGuard-main/
└── src/
    └── Final_Project/
        ├── python/              ← Bundled Python (~30 MB + packages)
        │   ├── python.exe
        │   ├── Lib/
        │   │   └── site-packages/  ← All dependencies here
        │   └── ...
        ├── trained_models/      ← Your trained models
        │   ├── Model1/
        │   └── Model2/
        ├── api_server/          ← API server
        │   ├── app.py
        │   └── start_server.bat
        ├── chrome_extension/    ← Chrome extension
        └── ...
```

## ⚠️ Important Notes

1. **First run may be slow** - Models load into memory (takes 30-60 seconds)
2. **Keep server window open** - Don't close it while using extension
3. **GPU not required** - Models work on CPU (slower but works)
4. **Internet not needed** - Everything is bundled!

## 🔧 Troubleshooting

### "Python not found"
- Make sure `python\` folder is in `Final_Project\`
- Check that `python\python.exe` exists

### "Models not found"
- Make sure `trained_models\Model1\` and `trained_models\Model2\` exist
- Train models first if missing

### "Port 5000 already in use"
- Close other applications using port 5000
- Or change port in `api_server\app.py`

### "Dependencies missing"
- Run `INSTALL_ALL_DEPENDENCIES.bat` again
- Make sure installation completed successfully

## ✅ Verification Checklist

Before copying to flash drive, verify:
- [ ] Python bundled in `python\` folder
- [ ] All dependencies installed (test with `python\python.exe -c "import torch; import tensorflow"`)
- [ ] Models trained and in `trained_models\` folder
- [ ] API server starts: `api_server\start_server.bat`
- [ ] Chrome extension loads without errors

## 🎉 That's It!

Once setup is complete, you have a **fully portable** PhishGuard that works on any Windows computer from a flash drive - no installation needed!

