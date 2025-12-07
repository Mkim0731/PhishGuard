# Quick Start Guide - PhishGuard

## 🚀 For Users Who Just Want to Get It Running

### Prerequisites Checklist

Before starting, make sure you have:

- [ ] **NVIDIA GPU** (required for training)
- [ ] **Python 3.8 or 3.9** installed
- [ ] **CUDA toolkit** installed
- [ ] **20GB free disk space**

---

## ⚡ 5-Minute Setup

### 1. Install Python Dependencies

```bash
# Navigate to project folder
cd Final_Project

# Install Model 1 dependencies
cd Model1_EmailContent
pip install -r requirements.txt

# Install Model 2 dependencies
cd ../Model2_URLDetection/Source_Code
pip install -r requirements.txt

# Install API dependencies (when created)
cd ../..
pip install flask flask-cors
```

### 2. Extract Datasets

```bash
cd Model1_EmailContent
# Extract archive.zip (if not already extracted)
python -c "import zipfile; zipfile.ZipFile('archive.zip').extractall('.')"
```

### 3. Verify GPU

```bash
# Check PyTorch GPU
python -c "import torch; print('PyTorch GPU:', torch.cuda.is_available())"

# Check TensorFlow GPU
python -c "import tensorflow as tf; print('TensorFlow GPU:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

Both should return `True` or show GPU devices.

### 4. Train Models

```bash
cd Final_Project
python train_both_models.py
```

**Time**: ~2-3 hours (one-time only)

### 5. Test It Works

```bash
# Test Model 1
cd Model1_EmailContent
python -c "from phishing_detection_system import PhishingDetector; d = PhishingDetector(require_cuda=False); d.load_model('../trained_models/Model1'); print(d.predict('Urgent: Your account is compromised!'))"

# Test Model 2
cd ../Model2_URLDetection/Source_Code
python url_predictor.py "https://test-url.com"
```

---

## 🎯 That's It!

After training completes:
- Models are saved in `trained_models/`
- No need to retrain
- Ready to use API server and Chrome extension

---

## ❌ Common Issues

**"CUDA not available"**
→ Install CUDA toolkit and PyTorch with CUDA support

**"Dataset not found"**
→ Extract `archive.zip` in `Model1_EmailContent/` folder

**"Module not found"**
→ Run `pip install -r requirements.txt` in each model folder

---

## 📖 Need More Help?

See `SETUP_INSTRUCTIONS.md` for detailed guide.

---

**Ready?** Start with Step 1 above! 🚀






