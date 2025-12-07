# PhishGuard: Code Architecture and System Design Report

## Executive Summary

PhishGuard is a dual-model machine learning system for phishing email detection, consisting of four major components: two independent ML models, a RESTful API server, and a Chrome browser extension. The system employs a client-server architecture where the extension acts as the user interface, the API server orchestrates model inference, and both models work in parallel to analyze email content and URLs.

---

## 1. System Architecture Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Chrome Browser Extension                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  popup.html  │  │  popup.js    │  │ content.js   │     │
│  │  (UI)        │  │  (Logic)     │  │ (Extraction) │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP POST /predict-combined
                            │ (email_text, urls[])
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Flask API Server (app.py)                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  /predict-combined endpoint                           │  │
│  │  - Receives email text and URLs                       │  │
│  │  - Orchestrates dual-model analysis                   │  │
│  │  - Combines results with weighted scoring             │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                 │
│          ┌─────────────────┴─────────────────┐             │
│          │                                     │             │
│          ▼                                     ▼             │
│  ┌──────────────┐                    ┌──────────────┐       │
│  │   Model 1    │                    │   Model 2    │       │
│  │  (Email)     │                    │   (URLs)     │       │
│  │  DistilBERT  │                    │     CNN      │       │
│  └──────────────┘                    └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ JSON Response
                            │ {email_result, url_results, 
                            │  is_phishing, combined_confidence}
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Chrome Extension (Display)                │
│  - Threat Score (0-100%)                                     │
│  - Email Content Analysis                                    │
│  - URL Analysis                                              │
│  - Warning Box (if score > 50%)                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Component Interaction Flow

1. **User Input**: User pastes email or clicks "Load from Page"
2. **Content Extraction**: `content.js` extracts email content from webpage
3. **URL Extraction**: `popup.js` extracts URLs using regex
4. **API Request**: Extension sends POST request to `/predict-combined`
5. **Model 1 Analysis**: API calls `PhishingDetector.predict()` on email text
6. **Model 2 Analysis**: API calls `URLPhishingDetector.predict_url()` on each URL
7. **Result Combination**: API combines results using weighted average
8. **Response**: JSON response sent back to extension
9. **Display**: Extension displays threat score, probabilities, and indicators

---

## 2. Project File Structure

```
Final_Project/
│
├── Model1_EmailContent/              # Model 1: Email Content Detection
│   ├── phishing_detection_system.py # Main class: PhishingDetector
│   │                                 # - Training logic
│   │                                 # - Model architecture (DistilBERT)
│   │                                 # - predict() method
│   │                                 # - load_model() method
│   ├── predict_email.py             # Standalone prediction script
│   ├── generate_visualizations.py   # Training metrics visualization
│   ├── *.csv                        # Training datasets (6 files)
│   └── requirements.txt             # PyTorch, transformers, etc.
│
├── Model2_URLDetection/             # Model 2: URL Detection
│   └── Source_Code/
│       ├── reproduction.py          # Main training script
│       │                            # - CNN architecture
│       │                            # - Data loading
│       │                            # - Training loop
│       ├── url_predictor.py         # Prediction class: URLPhishingDetector
│       │                            # - load_model() method
│       │                            # - predict_url() method
│       └── requirements.txt         # TensorFlow, Keras, etc.
│
├── trained_models/                  # Saved Model Files (After Training)
│   ├── Model1/                     # DistilBERT model files
│   │   ├── config.json           # Model configuration
│   │   ├── pytorch_model.bin     # Model weights
│   │   ├── tokenizer files       # Tokenizer for text processing
│   │   └── ...
│   └── Model2/                    # CNN model files
│       ├── model.h5              # Keras/TensorFlow model
│       └── tokenizer.pkl         # URL tokenizer
│
├── api_server/                     # RESTful API Server
│   ├── app.py                     # Main Flask application
│   │                              # - Flask app initialization
│   │                              # - Model loading (lazy loading)
│   │                              # - API endpoints
│   │                              # - Result combination logic
│   ├── start_server.bat           # Windows startup script
│   ├── requirements.txt           # Flask, flask-cors, psutil
│   └── HOW_TO_RUN_SERVER.md       # Server documentation
│
├── chrome_extension/               # Browser Extension
│   ├── manifest.json              # Extension configuration
│   │                              # - Permissions
│   │                              # - Content scripts
│   │                              # - Background worker
│   ├── popup.html                 # Extension UI structure
│   ├── popup.css                  # Styling
│   ├── popup.js                   # Main extension logic
│   │                              # - Email parsing
│   │                              # - API communication
│   │                              # - Result display
│   │                              # - Scoring calculation
│   ├── content.js                 # Content script (injected into pages)
│   │                              # - Email extraction from Gmail/Outlook
│   │                              # - Page content extraction
│   └── background.js               # Service worker
│                                  # - API health checks
│
├── train_both_models.py            # Master Training Script
│                                  # - Orchestrates training of both models
│                                  # - Handles user selection
│                                  # - Saves models to trained_models/
│
└── [Documentation Files]          # Various .md files for setup, testing, etc.
```

---

## 3. Core Components and Their Functions

### 3.1 Model 1: Email Content Detection (`Model1_EmailContent/`)

#### 3.1.1 Main File: `phishing_detection_system.py`

**Class: `PhishingDetector`**

**Key Methods:**

1. **`__init__()`**
   - Initializes DistilBERT tokenizer
   - Detects and configures GPU/CPU device
   - Sets up CUDA optimizations for training

2. **`clean_text()`**
   - Preprocesses email text
   - Removes HTML tags
   - Normalizes URLs to `[URL]`
   - Replaces emails with `[EMAIL]`
   - Normalizes whitespace

3. **`load_and_preprocess_data()`**
   - Loads 6 CSV datasets (CEAS_08, Enron, Ling, Nazario, Nigerian_Fraud, SpamAssasin)
   - Combines subject + body
   - Applies text cleaning
   - Standardizes labels (0=legitimate, 1=phishing)

4. **`train_model()`**
   - Initializes DistilBERTForSequenceClassification
   - Configures training arguments (batch size, learning rate, etc.)
   - Trains using HuggingFace Trainer
   - Saves model to `trained_models/Model1/`

5. **`load_model(model_path)`**
   - Loads trained DistilBERT model from disk
   - Loads tokenizer (with fallback to base model)
   - Sets model to evaluation mode

6. **`predict(email_text)`**
   - Cleans and tokenizes email text
   - Runs inference through DistilBERT model
   - Returns: `{is_phishing, phishing_probability, legitimate_probability, confidence}`

**Technology Stack:**
- **Framework**: PyTorch
- **Model**: DistilBERT (from HuggingFace Transformers)
- **Training**: HuggingFace Trainer API
- **Device**: CUDA GPU (RTX 4070 SUPER optimized)

---

### 3.2 Model 2: URL Detection (`Model2_URLDetection/Source_Code/`)

#### 3.2.1 Training File: `reproduction.py`

**Functions:**

1. **`load_data(path)`**
   - Loads URL dataset in chunks (memory-efficient)
   - Processes tab-separated values (label, URL)
   - Returns numpy arrays

2. **`create_model()`**
   - Defines CNN architecture:
     - Embedding layer
     - Conv1D layers (character-level)
     - MaxPooling layers
     - Dense layers
   - Compiles with Adam optimizer
   - Enables mixed precision (FP16) for GPU

3. **Training Loop**
   - Trains on URL dataset
   - Saves model to `trained_models/Model2/model.h5`
   - Saves tokenizer to `trained_models/Model2/tokenizer.pkl`

#### 3.2.2 Prediction File: `url_predictor.py`

**Class: `URLPhishingDetector`**

**Key Methods:**

1. **`__init__()`**
   - Initializes paths to model and tokenizer
   - Auto-detects paths if not provided

2. **`load_model()`**
   - Loads Keras/TensorFlow model from `model.h5`
   - Loads tokenizer from `tokenizer.pkl`

3. **`preprocess_url(url)`**
   - Converts URL to sequence using tokenizer
   - Pads/truncates to MAX_LEN (200 characters)

4. **`predict_url(url)`**
   - Preprocesses URL
   - Runs inference through CNN model
   - Returns: `{is_phishing, phishing_probability, legitimate_probability, confidence}`
   - **Threshold**: > 70% for `is_phishing` (refined)

**Technology Stack:**
- **Framework**: TensorFlow/Keras
- **Model**: CNN (Character-level)
- **Training**: Custom training loop with mixed precision
- **Device**: CUDA GPU (optimized for RTX 4070 SUPER)

---

### 3.3 API Server (`api_server/app.py`)

#### 3.3.1 Architecture

**Flask Application Structure:**

```python
app = Flask(__name__)
CORS(app)  # Enable cross-origin requests for Chrome extension

# Global model variables (lazy loaded)
model1 = None  # PhishingDetector instance
model2 = None  # URLPhishingDetector instance
```

#### 3.3.2 Key Functions

1. **`load_model1_lazy()`**
   - **Purpose**: Lazy loading of Model 1 (only when needed)
   - **Process**:
     - Checks if model already loaded
     - Imports `PhishingDetector` from `Model1_EmailContent`
     - Loads model from `trained_models/Model1/`
     - Verifies model loaded correctly
   - **Thread-safe**: Uses locks to prevent concurrent loading
   - **Memory Management**: Garbage collection before/after loading

2. **`load_model2_lazy()`**
   - **Purpose**: Lazy loading of Model 2 (only when needed)
   - **Process**:
     - Checks if model already loaded
     - Imports `URLPhishingDetector` from `Model2_URLDetection/Source_Code`
     - Loads model from `trained_models/Model2/`
   - **Thread-safe**: Uses locks

3. **`check_memory()`**
   - Monitors available system memory
   - Logs warnings if memory is low
   - Helps prevent paging file errors

#### 3.3.3 API Endpoints

**1. `GET /health`**
- **Purpose**: Health check and status
- **Returns**:
  ```json
  {
    "status": "healthy",
    "model1_loaded": true/false,
    "model2_loaded": true/false,
    "memory": {...}
  }
  ```

**2. `POST /predict-email`**
- **Purpose**: Analyze email content only
- **Request**: `{"email_text": "..."}`
- **Process**:
  1. Lazy load Model 1
  2. Call `model1.predict(email_text)`
  3. Apply keyword adjustment (1 keyword = max 25%)
  4. Return probabilities
- **Returns**: `{is_phishing, phishing_probability, legitimate_probability, confidence}`

**3. `POST /predict-url`**
- **Purpose**: Analyze single URL
- **Request**: `{"url": "https://..."}`
- **Process**:
  1. Lazy load Model 2
  2. Call `model2.predict_url(url)`
  3. Return probabilities
- **Returns**: `{is_phishing, phishing_probability, legitimate_probability, confidence}`

**4. `POST /predict-combined`** ⭐ **Main Endpoint**
- **Purpose**: Combined analysis using both models
- **Request**: 
  ```json
  {
    "email_text": "Subject: ... Body: ...",
    "urls": ["https://url1.com", "https://url2.com"]
  }
  ```
- **Process**:
  1. **Email Analysis**:
     - Lazy load Model 1
     - Call `model1.predict(email_text)`
     - Count suspicious keywords
     - Apply keyword adjustment (1 keyword → max 25%)
     - Store result in `email_result`
  
  2. **URL Analysis**:
     - Lazy load Model 2
     - For each URL: call `model2.predict_url(url)`
     - Store results in `url_results[]`
  
  3. **Combined Decision**:
     - Email threshold: > 65% for suspicious
     - URL threshold: > 70% AND at least 2 URLs (or 1 URL > 85%)
     - Mark as phishing if:
       - (Email > 65% AND URLs suspicious) OR
       - Email > 85% OR
       - URL > 90%
  
  4. **Scoring**:
     - Weighted average: (Email × 60%) + (URL × 40%)
     - If multiple URLs: average of top 2
     - Store in `combined_confidence`
  
- **Returns**:
  ```json
  {
    "email_result": {
      "is_phishing": false,
      "phishing_probability": 0.25,
      "legitimate_probability": 0.75,
      "confidence": 0.75
    },
    "url_results": [
      {
        "url": "https://...",
        "is_phishing": false,
        "phishing_probability": 0.60,
        "legitimate_probability": 0.40
      }
    ],
    "is_phishing": false,
    "combined_confidence": 0.39
  }
  ```

**Technology Stack:**
- **Framework**: Flask (Python web framework)
- **CORS**: flask-cors (for Chrome extension)
- **Memory Monitoring**: psutil
- **Threading**: Thread-safe model loading

---

### 3.4 Chrome Extension (`chrome_extension/`)

#### 3.4.1 File Structure and Functions

**1. `manifest.json`**
- **Purpose**: Extension configuration (Manifest V3)
- **Key Settings**:
  - Permissions: `activeTab`, `storage`, `scripting`
  - Host permissions: Gmail, Outlook, localhost:5000, all URLs
  - Content scripts: Injected into Gmail/Outlook pages
  - Background: Service worker for API health checks
  - Action: Popup UI when extension icon clicked

**2. `popup.html`**
- **Purpose**: User interface structure
- **Components**:
  - Email input textarea
  - "Load from Page" button
  - "Detect Phishing and Spam" button
  - Results section (threat score, probabilities, URLs)
  - Warning box
  - "Show More Details" section
  - "Show Parsed Content" section

**3. `popup.css`**
- **Purpose**: Styling for extension UI
- **Features**: Modern gradient design, responsive layout, color-coded scores

**4. `popup.js`** ⭐ **Main Extension Logic**

**Key Functions:**

1. **`parseEmail(emailContent)`**
   - **Purpose**: Extract subject and body from raw email
   - **Process**:
     - Identifies MIME boundaries
     - Handles multipart emails
     - Decodes base64 and quoted-printable encoding
     - Extracts HTML text content
     - Removes email headers
   - **Returns**: `{subject, body}`

2. **`extractURLs(emailContent)`**
   - **Purpose**: Find all URLs in email
   - **Method**: Regex pattern matching
   - **Returns**: Array of URL strings

3. **`loadEmailFromPage()`**
   - **Purpose**: Extract email from current webpage
   - **Process**:
     - Injects `content.js` into page
     - Calls `extractEmailContent()` from content script
     - Populates email input field
   - **Works on**: Gmail, Outlook, or any webpage

4. **`detectPhishing(emailText, urls)`**
   - **Purpose**: Send analysis request to API
   - **Process**:
     - POST request to `http://localhost:5000/predict-combined`
     - Sends `{email_text, urls}`
     - 30-second timeout
     - Returns JSON response
   - **Error Handling**: Catches network errors, timeouts

5. **`handleDetect()`**
   - **Purpose**: Main detection handler (called when button clicked)
   - **Process**:
     1. Gets email content from textarea
     2. Extracts URLs
     3. Shows loading state
     4. Calls `detectPhishing()`
     5. Calls `displayResults()`
     6. Handles errors

6. **`displayResults(result, urls, ...)`**
   - **Purpose**: Display analysis results in UI
   - **Process**:
     - Calculates threat score (weighted average)
     - Updates threat score circle
     - Displays email probabilities
     - Displays URL scores
     - Shows/hides warning box (if score > 50%)
     - Analyzes and displays indicators
     - Shows parsed email content
   - **Scoring**:
     ```javascript
     Threat Score = (Email_Prob × 60%) + (URL_Prob × 40%)
     ```

7. **`analyzeEmailIndicators(...)`**
   - **Purpose**: Generate explainable AI indicators
   - **Checks**:
     - Suspicious keywords count
     - URL patterns (short links, HTTP)
     - Urgency indicators
     - Sender analysis
     - Grammar/spelling
     - AI confidence levels
   - **Returns**: Array of indicator objects

8. **`highlightSuspiciousContent(text)`**
   - **Purpose**: Highlight suspicious keywords and URLs in parsed content
   - **Method**: Regex replacement with HTML `<mark>` tags

**5. `content.js`**
- **Purpose**: Content script injected into web pages
- **Functions**:
  - `extractGmailEmail()`: Extracts email from Gmail page
  - `extractOutlookEmail()`: Extracts email from Outlook page
  - `extractGenericEmail()`: Extracts content from any page
- **Returns**: Full page HTML or email content

**6. `background.js`**
- **Purpose**: Service worker for background tasks
- **Functions**:
  - `checkAPIHealth()`: Checks if API server is running
  - Listens for messages from content scripts

**Technology Stack:**
- **Manifest**: Chrome Extension Manifest V3
- **Languages**: HTML, CSS, JavaScript
- **APIs**: Chrome Extension APIs (tabs, storage, scripting)
- **Communication**: HTTP REST API (fetch)

---

## 4. Data Flow and Integration

### 4.1 Complete Request-Response Flow

```
┌──────────────────────────────────────────────────────────────┐
│ STEP 1: User Interaction                                     │
│ User pastes email or clicks "Load from Page"                  │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 2: Content Extraction (popup.js)                        │
│ - parseEmail() extracts subject and body                     │
│ - extractURLs() finds all URLs using regex                    │
│ - Content stored in variables                                 │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 3: API Request (popup.js → app.py)                      │
│ POST http://localhost:5000/predict-combined                  │
│ Body: {                                                       │
│   "email_text": "Subject: ... Body: ...",                    │
│   "urls": ["https://url1.com", "https://url2.com"]          │
│ }                                                             │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 4: Model 1 Analysis (app.py → phishing_detection_system)│
│ - load_model1_lazy() loads DistilBERT model                   │
│ - model1.predict(email_text) runs inference                   │
│ - Returns: {phishing_prob: 0.75, legit_prob: 0.25}          │
│ - Keyword adjustment applied (if 1 keyword → max 25%)        │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 5: Model 2 Analysis (app.py → url_predictor)            │
│ - load_model2_lazy() loads CNN model                         │
│ - For each URL: model2.predict_url(url)                      │
│ - Returns: [{url: "...", phishing_prob: 0.85}, ...]         │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 6: Result Combination (app.py)                           │
│ - Combined decision logic applied                             │
│ - Weighted average calculated:                               │
│   (Email × 60%) + (URL × 40%)                               │
│ - is_phishing flag determined                                │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 7: API Response (app.py → popup.js)                     │
│ JSON: {                                                      │
│   email_result: {...},                                       │
│   url_results: [...],                                        │
│   is_phishing: true/false,                                   │
│   combined_confidence: 0.73                                  │
│ }                                                             │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 8: Display Results (popup.js)                           │
│ - displayResults() updates UI                                │
│ - Threat score displayed (0-100%)                            │
│ - Probabilities shown                                        │
│ - Warning box shown if score > 50%                          │
│ - Indicators displayed                                       │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 Model Loading Strategy (Lazy Loading)

**Why Lazy Loading?**
- Models are large (Model 1: ~250MB, Model 2: ~50MB)
- Reduces startup memory footprint
- Models only load when first prediction is requested

**Implementation:**
```python
# Global variables
model1 = None
model2 = None
model1_lock = threading.Lock()  # Thread safety

def load_model1_lazy():
    global model1
    with model1_lock:
        if model1 is not None:
            return model1  # Already loaded
        
        # Load model...
        model1 = PhishingDetector()
        model1.load_model(path)
        return model1
```

**Benefits:**
- Server starts quickly
- Memory used only when needed
- Thread-safe for concurrent requests

---

## 5. Key Design Decisions

### 5.1 Dual-Model Architecture

**Rationale:**
- **Separation of Concerns**: Email content and URLs require different analysis techniques
- **Specialization**: Each model optimized for its domain
- **Accuracy**: Combined approach achieves higher accuracy than single model

**Implementation:**
- Model 1: Transformer-based (understands context, semantics)
- Model 2: CNN-based (pattern recognition for URLs)

### 5.2 Weighted Average Scoring

**Formula:**
```
Threat Score = (Email_Probability × 60%) + (URL_Probability × 40%)
```

**Rationale:**
- Email content is more reliable indicator (60% weight)
- URLs can be misleading (40% weight)
- More nuanced than taking maximum
- Prevents single indicator from dominating

### 5.3 Keyword-Based Adjustment

**Logic:**
- 1 suspicious keyword → Max 25% phishing probability
- Prevents false positives from single words
- Legitimate emails may contain words like "urgent" or "verify"

**Implementation:**
```python
if keyword_count == 1:
    phishing_prob = min(phishing_prob, 0.25)  # Cap at 25%
```

### 5.4 Refined Thresholds

**Email Threshold**: 65% (was 50%)
- Requires stronger evidence before flagging
- Reduces false positives

**URL Threshold**: 70% (was 50%)
- URLs need to be clearly suspicious
- Prevents false positives from benign URLs

**Combined Decision**: Requires BOTH email AND URLs suspicious (or very high individual)
- Prevents false positives from single indicators
- Only very high scores (>85% email or >90% URL) trigger alone

---

## 6. Technology Stack Summary

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Model 1** | PyTorch, DistilBERT (Transformers) | Email content analysis |
| **Model 2** | TensorFlow/Keras, CNN | URL pattern detection |
| **API Server** | Flask, Flask-CORS | RESTful API, model orchestration |
| **Extension** | HTML, CSS, JavaScript, Chrome APIs | User interface, email extraction |
| **Training** | HuggingFace Trainer, Custom training loops | Model training |
| **Deployment** | Python embeddable package | Portable deployment |

---

## 7. File Dependencies and Connections

### 7.1 Training Phase

```
train_both_models.py
    │
    ├──→ Model1_EmailContent/phishing_detection_system.py
    │       └──→ Trains DistilBERT model
    │       └──→ Saves to: trained_models/Model1/
    │
    └──→ Model2_URLDetection/Source_Code/reproduction.py
            └──→ Trains CNN model
            └──→ Saves to: trained_models/Model2/
```

### 7.2 Runtime Phase

```
Chrome Extension (popup.js)
    │
    └──→ HTTP POST → api_server/app.py
                │
                ├──→ Model1_EmailContent/phishing_detection_system.py
                │       └──→ PhishingDetector.predict()
                │       └──→ Loads from: trained_models/Model1/
                │
                └──→ Model2_URLDetection/Source_Code/url_predictor.py
                        └──→ URLPhishingDetector.predict_url()
                        └──→ Loads from: trained_models/Model2/
```

### 7.3 Import Structure

**API Server (`app.py`) imports:**
```python
sys.path.insert(0, 'Model1_EmailContent/')
from phishing_detection_system import PhishingDetector

sys.path.insert(0, 'Model2_URLDetection/Source_Code/')
from url_predictor import URLPhishingDetector
```

**Model 1 imports:**
```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
```

**Model 2 imports:**
```python
from tensorflow.keras.models import load_model
import tensorflow as tf
```

---

## 8. Error Handling and Robustness

### 8.1 Model Loading Errors
- **Paging File Issues**: Clear error messages with solutions
- **Memory Errors**: Memory checks before loading
- **File Not Found**: Helpful error messages pointing to training requirement

### 8.2 API Errors
- **Model Not Loaded**: Returns error JSON instead of crashing
- **Invalid Input**: Validates request body, returns 400 errors
- **Prediction Errors**: Catches exceptions, returns default values

### 8.3 Extension Errors
- **API Timeout**: 30-second timeout prevents hanging
- **Network Errors**: User-friendly error messages
- **Empty Content**: Validates input before sending

---

## 9. Performance Optimizations

### 9.1 GPU Acceleration
- **Model 1**: CUDA-enabled PyTorch, mixed precision (FP16)
- **Model 2**: TensorFlow GPU, mixed precision, XLA compilation
- **Training**: Optimized batch sizes for RTX 4070 SUPER (12.9 GB VRAM)

### 9.2 Memory Management
- **Lazy Loading**: Models load on-demand
- **Garbage Collection**: Explicit GC before/after model loading
- **Chunked Data Loading**: Model 2 loads data in chunks

### 9.3 Caching
- **Model Caching**: Models loaded once, reused for all requests
- **Thread Safety**: Locks prevent concurrent loading

---

## 10. Security Considerations

### 10.1 Extension Security
- **Content Security Policy**: No inline scripts
- **Permissions**: Minimal required permissions
- **Local API**: Only connects to localhost:5000

### 10.2 API Security
- **CORS**: Enabled only for Chrome extension
- **Input Validation**: Validates all inputs
- **Error Messages**: Don't expose sensitive information

---

## 11. Conclusion

PhishGuard employs a **modular, client-server architecture** where:

1. **Two independent ML models** specialize in different aspects (email content vs. URLs)
2. **A Flask API server** orchestrates model inference and combines results
3. **A Chrome extension** provides the user interface and handles email extraction
4. **Lazy loading** optimizes memory usage
5. **Weighted scoring** provides nuanced threat assessment
6. **Refined thresholds** reduce false positives while maintaining detection

The system is designed for **portability** (bundled Python), **robustness** (comprehensive error handling), and **usability** (simple startup scripts, clear documentation).

---

## Appendix: Key File Functions Reference

| File | Primary Function |
|------|-----------------|
| `phishing_detection_system.py` | Model 1 training and prediction |
| `reproduction.py` | Model 2 training |
| `url_predictor.py` | Model 2 prediction |
| `app.py` | API server, model orchestration |
| `popup.js` | Extension logic, UI updates |
| `content.js` | Email extraction from web pages |
| `train_both_models.py` | Master training orchestrator |

---

*This document provides a comprehensive overview of the PhishGuard codebase architecture for academic and professional reporting purposes.*

