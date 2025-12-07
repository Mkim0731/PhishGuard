# PhishGuard - GitHub Repository

This folder contains the organized project files ready for GitHub upload.

**Total Size:** ~0.56 MB (down from ~1.4 GB after excluding large files)

## What's Included

- ✅ All source code (.py files)
- ✅ Configuration files (.json, .txt, .md)
- ✅ Chrome extension files (including icons)
- ✅ API server code
- ✅ Documentation (all .md files)
- ✅ Setup scripts (.bat, .sh, .ps1)
- ✅ Test email samples (small text files)
- ✅ Small visualization images (PNG files)

## What's Excluded

- ❌ Trained model files (.pt, .h5, .pkl, .pth, .safetensors, .bin) - too large
- ❌ Dataset files (.csv) - too large
- ❌ Large dataset text files (>1MB in dataset/archive folders)
- ❌ Embedded Python installation (Python/ folder) - too large
- ❌ Archive folders with pre-trained models
- ❌ Log files (.log)
- ❌ Cache files (__pycache__)
- ❌ Temporary model files (phishing_model_temp/)
- ❌ Binary files (.pyd, .dll, .exe)

## File Structure

```
sorted/
├── api_server/              # Flask API server
├── chrome_extension/        # Chrome extension files
├── Model1_EmailContent/     # Email content detection model (source code)
├── Model2_URLDetection/     # URL detection model (source code)
├── test_emails/            # Sample test emails
├── *.py                     # Main training and test scripts
├── *.md                     # Documentation files
├── *.bat, *.sh, *.ps1      # Setup scripts
└── .gitignore              # Git ignore rules
```

## Next Steps for GitHub Upload

1. **Review the files** in this folder to ensure everything needed is included
2. **Navigate to sorted folder**: `cd sorted`
3. **Initialize git**: `git init`
4. **Add files**: `git add .`
5. **Commit**: `git commit -m "Initial commit: PhishGuard phishing detection system"`
6. **Create GitHub repository** (on GitHub website)
7. **Add remote**: `git remote add origin https://github.com/yourusername/your-repo.git`
8. **Push**: `git push -u origin main`

## Important Notes

### For Users Cloning This Repository

Users will need to:

1. **Install Python and dependencies**
   - Python 3.8+
   - Install requirements: `pip install -r REQUIREMENTS.txt`

2. **Train the models** (or download pre-trained models separately)
   - Model 1: Run `python Model1_EmailContent/phishing_detection_system.py`
   - Model 2: Run `python Model2_URLDetection/Source_Code/reproduction.py`
   - Or use the combined script: `python train_both_models.py`

3. **Download datasets separately** if needed for training
   - Datasets are not included due to size
   - Users should obtain datasets from original sources

4. **Models will be saved to** `trained_models/` after training

### Large Files

If you need to include large files (models, datasets), consider:
- Using Git LFS (Large File Storage): `git lfs install`
- Hosting large files separately (Google Drive, Dropbox, etc.)
- Providing download links in the README

## .gitignore

The `.gitignore` file is already configured to exclude:
- Model files (*.pt, *.h5, *.pkl, *.pth, *.safetensors)
- Dataset files (*.csv, large *.txt)
- Python cache (__pycache__)
- Log files
- Binary files
- And more...

You can review and modify `.gitignore` if needed.
