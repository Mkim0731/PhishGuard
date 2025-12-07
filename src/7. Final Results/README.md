# PhishGuard - Final Project

## 🎯 Project Overview

This is the **final integrated project** combining two ML models for comprehensive phishing detection:

1. **Model 1 (Email Content)**: DistilBERT model analyzing email text content
2. **Model 2 (URL Detection)**: CNN model analyzing URLs in emails

Both models work together to provide robust phishing detection for a Chrome extension.

---

## 📁 Project Structure

```
Final_Project/
├── Model1_EmailContent/          # Email content detection (DistilBERT)
│   ├── phishing_detection_system.py  # Main training script
│   ├── archive.zip               # Datasets (extracted)
│   ├── *.csv                     # Training datasets
│   └── ...
│
├── Model2_URLDetection/          # URL phishing detection (CNN)
│   └── Source_Code/
│       ├── reproduction.py       # Main training script
│       ├── url_predictor.py      # Prediction module
│       └── ...
│
├── trained_models/               # Trained models (after training)
│   ├── Model1/                   # DistilBERT model files
│   └── Model2/                   # CNN model files
│
├── api_server/                   # API server (to be created)
├── chrome_extension/             # Chrome extension (to be created)
│
├── train_both_models.py          # Master training script
├── TRAINING_GUIDE.md             # Training instructions
└── README.md                     # This file
```

---

## 🚀 Quick Start

**New to this project?** Start here:
1. Read `SETUP_INSTRUCTIONS.md` for complete setup guide
2. Or see `QUICK_START.md` for fast setup

### Step 1: Train Both Models (One-Time)

**Option A: Train both at once**
```bash
cd "N:\Senior Project\PhishGuard-main\src\Final_Project"
python train_both_models.py
```

**Option B: Train individually**
- Model 1: `cd Model1_EmailContent && python phishing_detection_system.py`
- Model 2: `cd Model2_URLDetection/Source_Code && python reproduction.py`

**Training Time:**
- Model 1: ~45 minutes (requires GPU)
- Model 2: ~1.5 hours

**After training:** Models are saved to `trained_models/` and can be reused without retraining.

### Step 2: Test Predictions

**Test Model 1:**
```python
from Model1_EmailContent.phishing_detection_system import PhishingDetector

detector = PhishingDetector(require_cuda=False)
detector.load_model('../trained_models/Model1')
result = detector.predict('Subject: Urgent Body: Your account has been compromised...')
print(result)
```

**Test Model 2:**
```bash
cd Model2_URLDetection/Source_Code
python url_predictor.py "https://suspicious-url.com"
```

---

## 📊 Model Details

### Model 1: Email Content Detection
- **Architecture**: DistilBERT (Transformer)
- **Framework**: PyTorch
- **Input**: Email text (subject + body combined)
- **Accuracy**: ~92%
- **Model Size**: ~250MB
- **Training Time**: ~45 min (GPU required)

### Model 2: URL Detection
- **Architecture**: CNN (Convolutional Neural Network)
- **Framework**: TensorFlow/Keras
- **Input**: URL strings (character-level)
- **Accuracy**: ~98%
- **Model Size**: ~1-2MB
- **Training Time**: ~1.5 hours

---

## ✅ Current Status

### Completed ✅
- [x] Project structure organized
- [x] Both models copied and organized
- [x] Datasets extracted for Model 1
- [x] Model 1: Added `load_model()` and `predict()` methods
- [x] Model 2: Added `load_model()` and `predict_url()` methods
- [x] Training scripts modified to save to `trained_models/`
- [x] Master training script created
- [x] Training guide created

### Next Steps ⏭️
- [ ] Train both models (run `train_both_models.py`)
- [ ] Create unified API server
- [ ] Build Chrome extension
- [ ] Integrate both models in extension

---

## 🔧 Requirements

### Model 1 Requirements:
- Python 3.8+
- CUDA GPU (for training)
- PyTorch with CUDA
- ~10GB disk space

### Model 2 Requirements:
- Python 3.8+
- TensorFlow 2.2.0+
- ~500MB disk space

### Installation:
```bash
# Model 1 dependencies
cd Model1_EmailContent
pip install -r requirements.txt

# Model 2 dependencies
cd Model2_URLDetection/Source_Code
pip install -r requirements.txt
```

---

## 📖 Documentation

### Setup & Installation:
- **SETUP_INSTRUCTIONS.md**: Complete setup guide for new users
- **QUICK_START.md**: Fast 5-minute setup guide
- **TRAINING_GUIDE.md**: Detailed training instructions

### Technical Documentation:
- **EXTENSION_ARCHITECTURE.md**: Architecture and performance analysis
- **PERFORMANCE_SUMMARY.md**: Performance metrics and expectations
- **Model1_EmailContent/README.md**: Model 1 documentation
- **Model1_EmailContent/ANALYSIS.md**: Deep analysis of Model 1
- **Model2_URLDetection/ANALYSIS.md**: Deep analysis of Model 2

---

## 🎯 Usage After Training

Once models are trained, you can:

1. **Load and use Model 1:**
```python
from Model1_EmailContent.phishing_detection_system import PhishingDetector

detector = PhishingDetector(require_cuda=False)  # CPU for inference
detector.load_model('trained_models/Model1')
result = detector.predict(email_text)
```

2. **Load and use Model 2:**
```python
from Model2_URLDetection.Source_Code.url_predictor import URLPhishingDetector

detector = URLPhishingDetector()
detector.load_model()
result = detector.predict_url(url_string)
```

3. **Combine both models:**
```python
# Check email content
email_result = model1.predict(email_text)

# Check all URLs in email
url_results = [model2.predict_url(url) for url in urls_in_email]

# Combined decision
is_phishing = email_result['is_phishing'] or any(r['is_phishing'] for r in url_results)
```

---

## 🚨 Important Notes

1. **Training is one-time**: After training, models are saved and can be reused
2. **Model 1 requires GPU**: Cannot train on CPU (too slow)
3. **Model 2 can use CPU**: Works without GPU (slower)
4. **Inference can use CPU**: Both models can run on CPU for predictions (slower but works)

---

## 📞 Next Steps

After training both models:

1. ✅ Models trained and saved
2. ⏭️ Create API server (`api_server/`)
3. ⏭️ Build Chrome extension (`chrome_extension/`)
4. ⏭️ Integrate both models
5. ⏭️ Test end-to-end

---

**Ready to train?** Run `python train_both_models.py` and grab a coffee! ☕

