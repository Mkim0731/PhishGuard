# PhishGuard - Complete Setup Instructions

## 📋 For School Project Submission / Other Users

This guide will help anyone set up and run the PhishGuard phishing detection system on their machine.

---

## 🎯 What This Project Does

PhishGuard is a dual-model phishing detection system that:
- Analyzes email content for phishing indicators (Model 1: DistilBERT)
- Analyzes URLs in emails for phishing patterns (Model 2: CNN)
- Works as a Chrome extension to protect users from phishing emails

---

## ⚠️ Prerequisites (REQUIRED)

### Hardware Requirements:

1. **NVIDIA GPU with CUDA support** (REQUIRED for training)
   - Minimum: Any CUDA-compatible GPU
   - Recommended: RTX 3060 or better
   - **Why**: Training on CPU would take days/weeks

2. **System Requirements**:
   - **RAM**: 8GB+ (16GB recommended)
   - **Storage**: 20GB+ free space
   - **OS**: Windows 10/11, Linux, or macOS (Windows recommended)

### Software Requirements:

1. **Python 3.8 or 3.9** (3.10+ may have compatibility issues)
   - Download from: https://www.python.org/downloads/
   - **Important**: Check "Add Python to PATH" during installation**

2. **NVIDIA GPU Drivers**
   - Download from: https://www.nvidia.com/drivers
   - Install latest drivers for your GPU

3. **CUDA Toolkit 11.8 or 12.1**
   - Download from: https://developer.nvidia.com/cuda-downloads
   - **Important**: Match CUDA version with PyTorch/TensorFlow versions

4. **Git** (optional, for cloning repository)
   - Download from: https://git-scm.com/downloads

---

## 📦 Step-by-Step Installation

### Step 1: Clone/Download the Project

**Option A: If you have the project folder:**
- Copy the entire `Final_Project` folder to your machine
- Navigate to it in terminal/command prompt

**Option B: If using Git:**
```bash
git clone <repository-url>
cd PhishGuard-main/src/Final_Project
```

### Step 2: Verify Python Installation

Open terminal/command prompt and check:

```bash
python --version
# Should show: Python 3.8.x or 3.9.x

pip --version
# Should show: pip version
```

If Python is not found, add it to PATH or reinstall Python.

### Step 3: Install Model 1 Dependencies (DistilBERT)

```bash
cd Model1_EmailContent
pip install -r requirements.txt
```

**If you get CUDA errors**, install PyTorch with CUDA manually:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verify GPU is detected:**
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Should output: `CUDA available: True` and your GPU name.

### Step 4: Install Model 2 Dependencies (CNN)

```bash
cd ../Model2_URLDetection/Source_Code
pip install -r requirements.txt
```

**If TensorFlow GPU is needed:**
```bash
pip install tensorflow-gpu==2.2.0
```

**Verify GPU is detected:**
```bash
python -c "import tensorflow as tf; print('GPU devices:', tf.config.list_physical_devices('GPU'))"
```

### Step 5: Extract Datasets

```bash
cd ../../Model1_EmailContent
# If archive.zip exists, extract it
python -c "import zipfile; zipfile.ZipFile('archive.zip').extractall('.')"
```

Or manually extract `archive.zip` to get the CSV files.

**Verify datasets exist:**
```bash
# Should see these files:
# CEAS_08.csv
# Enron.csv
# Ling.csv
# Nazario.csv
# Nigerian_Fraud.csv
# SpamAssasin.csv
```

---

## 🚀 Training the Models

### Option A: Train Both Models (Recommended)

```bash
cd Final_Project
python train_both_models.py
```

This will:
1. Check for GPU (will error if not available)
2. Train Model 1 (~45 minutes)
3. Train Model 2 (~1.5 hours)
4. Save models to `trained_models/` folder

**Total time**: ~2-3 hours

### Option B: Train Models Individually

**Train Model 1:**
```bash
cd Model1_EmailContent
python phishing_detection_system.py
```

**Train Model 2:**
```bash
cd Model2_URLDetection/Source_Code
python reproduction.py
```

### ⚠️ Important Training Notes:

- **GPU is REQUIRED**: Scripts will error if GPU not detected
- **Training is one-time**: Models are saved and can be reused
- **Don't interrupt**: Let training complete fully
- **Check disk space**: Models need ~500MB total

---

## ✅ Verify Models Are Trained

After training, check:

```bash
# Check Model 1
dir trained_models\Model1
# Should see: config.json, pytorch_model.bin, tokenizer files

# Check Model 2
dir trained_models\Model2
# Should see: model.h5, tokenizer.pkl
```

---

## 🧪 Test the Models

### Test Model 1 (Email Content):

```bash
cd Model1_EmailContent
python -c "from phishing_detection_system import PhishingDetector; import sys; sys.path.append('..'); d = PhishingDetector(require_cuda=False); d.load_model('../trained_models/Model1'); result = d.predict('Subject: Urgent Body: Your account has been compromised. Click here to verify.'); print(result)"
```

### Test Model 2 (URL Detection):

```bash
cd Model2_URLDetection/Source_Code
python url_predictor.py "https://fake-bank.com/verify"
```

---

## 🌐 Setting Up API Server (For Chrome Extension)

### Step 1: Install API Dependencies

```bash
cd Final_Project
pip install flask flask-cors
```

### Step 2: Run API Server

```bash
# API server will be created in next step
# For now, this is a placeholder
python api_server/app.py
```

**Note**: API server code will be created in the next development phase.

---

## 🔌 Setting Up Chrome Extension

### Step 1: Load Extension in Chrome

1. Open Chrome
2. Go to `chrome://extensions/`
3. Enable "Developer mode" (toggle in top-right)
4. Click "Load unpacked"
5. Select the `chrome_extension` folder

**Note**: Chrome extension code will be created in the next development phase.

### Step 2: Configure Extension

- Extension will connect to API server
- Default: `http://localhost:5000`
- Can be changed in extension settings

---

## 📁 Project Structure (What You Should Have)

```
Final_Project/
├── Model1_EmailContent/
│   ├── phishing_detection_system.py
│   ├── *.csv (6 dataset files)
│   └── requirements.txt
│
├── Model2_URLDetection/
│   └── Source_Code/
│       ├── reproduction.py
│       ├── url_predictor.py
│       └── requirements.txt
│
├── trained_models/
│   ├── Model1/          (after training)
│   └── Model2/          (after training)
│
├── api_server/          (to be created)
├── chrome_extension/    (to be created)
│
├── train_both_models.py
├── SETUP_INSTRUCTIONS.md (this file)
└── README.md
```

---

## 🔧 Troubleshooting

### Problem: "CUDA not available"

**Solutions:**
1. Check NVIDIA drivers are installed: `nvidia-smi` in terminal
2. Verify CUDA toolkit is installed
3. Reinstall PyTorch with CUDA:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Problem: "Dataset files not found"

**Solutions:**
1. Extract `archive.zip` in `Model1_EmailContent/` folder
2. Verify all 6 CSV files are present
3. Check file names match exactly (case-sensitive)

### Problem: "Out of memory" during training

**Solutions:**
1. Close other GPU applications
2. Reduce batch size in training scripts
3. Use a GPU with more VRAM

### Problem: "Module not found" errors

**Solutions:**
1. Install missing packages: `pip install <package-name>`
2. Check you're in the correct directory
3. Use virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate  # Linux/Mac
   pip install -r requirements.txt
   ```

### Problem: Training takes too long

**Solutions:**
1. Verify GPU is being used (check task manager)
2. Ensure CUDA is properly installed
3. Check GPU drivers are up to date

---

## 📝 Quick Start Checklist

- [ ] Python 3.8/3.9 installed
- [ ] NVIDIA GPU drivers installed
- [ ] CUDA toolkit installed
- [ ] Project folder downloaded/copied
- [ ] Model 1 dependencies installed (`pip install -r requirements.txt`)
- [ ] Model 2 dependencies installed
- [ ] Datasets extracted (6 CSV files)
- [ ] GPU verified working (`python -c "import torch; print(torch.cuda.is_available())"`)
- [ ] Models trained (`python train_both_models.py`)
- [ ] Models verified in `trained_models/` folder
- [ ] API server installed and running
- [ ] Chrome extension loaded

---

## 🎓 For School Project Submission

### What to Include:

1. **Project folder** with all code
2. **Trained models** in `trained_models/` folder (or instructions to train)
3. **This setup guide** (`SETUP_INSTRUCTIONS.md`)
4. **README.md** with project overview
5. **Requirements files** for easy installation

### What to Document:

1. **System requirements** (GPU, RAM, etc.)
2. **Installation steps** (this guide)
3. **Training instructions**
4. **Usage examples**
5. **Known issues/limitations**

### Demo Instructions:

1. Show trained models exist
2. Demonstrate prediction on sample email
3. Show Chrome extension working
4. Explain architecture (dual-model approach)

---

## 📞 Getting Help

If you encounter issues:

1. **Check this guide** first
2. **Verify prerequisites** (GPU, CUDA, Python)
3. **Check error messages** carefully
4. **Review troubleshooting section**
5. **Check project documentation** (README.md, TRAINING_GUIDE.md)

---

## ⏱️ Time Estimates

- **Initial setup**: 30-60 minutes
- **Training Model 1**: ~45 minutes (GPU required)
- **Training Model 2**: ~1.5 hours (GPU recommended)
- **Total setup time**: ~3-4 hours (first time)

**After initial setup**: Models are saved, no retraining needed!

---

## ✅ Success Criteria

You know everything is working when:

1. ✅ `python train_both_models.py` completes without errors
2. ✅ `trained_models/Model1/` contains model files
3. ✅ `trained_models/Model2/` contains model files
4. ✅ Test predictions work (see "Test the Models" section)
5. ✅ API server starts without errors
6. ✅ Chrome extension loads and connects to API

---

## 🎯 Summary

**Minimum Requirements:**
- NVIDIA GPU with CUDA
- Python 3.8/3.9
- 20GB disk space
- 8GB RAM

**Steps:**
1. Install Python, CUDA, GPU drivers
2. Install dependencies
3. Extract datasets
4. Train models (one-time, ~2-3 hours)
5. Run API server
6. Load Chrome extension

**That's it!** After training, the system is ready to use.

---

**Good luck with your project!** 🚀






