# SIMILAR PROJECT ON A GOOGLE EXTENSION

## 1) https://github.com/shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques 

### 📦 Data Collection
Phishing URLs: 5,000 samples from PhishTank \
Legitimate URLs: 5,000 samples from the University of New Brunswick's dataset \
All data files are in the repository’s DataFiles folder.

### 🧪 Feature Extraction
17 features are extracted from URLs, categorized into: 
* Address bar-based features (9)
* Domain-based features (4)
* TML & JavaScript-based features (4)

Detailed feature extraction is documented in URL Feature Extraction.ipynb.

### 🤖 Machine Learning Models Used
* Decision Tree 
* Random Forest
* Multilayer Perceptrons 
* XGBoost
* Autoencoder
* Neural Network
* Support Vector Machines 

Training and evaluation are detailed in Phishing Website Detection_Models & Training.ipynb.

### ✅ Best Performing Model
XGBoost achieved the highest accuracy: 86.4% \
Saved model: XGBoostClassifier.pickle.dat, later implemented using a browser extension.

## 2) https://github.com/Click2Hack/Phishing-Email-Detection-Using-Machine-Learning?tab=readme-ov-file

### 🧰 Key Features
* Email Classification: Labels emails as "Phishing" or "Not Phishing".
* Text Preprocessing: Uses TF-IDF vectorization to convert email text into features.
* Model Training: Trains ML models (default: Random Forest) on labeled data.
* Prediction: Allows prediction of new emails based on trained model.
* Optional: Flask integration for web deployment.

### 🧪 Workflow
Step-by-step Process:
* Preprocess Data
* Run preprocess.py to vectorize emails and save processed data.
* Train the Model
* Run train.py to train and save the ML model as phishing_detector.pkl.
* Uses RandomForestClassifier by default.
* Make Predictions
* Run predict.py to test new email samples.
* Modify email_text to test custom messages.

### 🧱 Technologies Used
* Python
* Scikit-learn (ML models)
* Pandas & NumPy (data manipulation)
* Pickle (model serialization)
* TF-IDF Vectorization (NLP)
* Flask (optional, for deployment)

## 3) https://github.com/fennybz/Detecting-Phishing-Attack-using-ML-DL-Models

### 🗂️ Dataset

The following open-source email datasets were used:
* SpamAssassin
* Spam/Ham Dataset

Each dataset includes labeled phished and legitimate emails.

### 🔍 Feature Extraction
Emails were processed using NLP tools to extract relevant features.

These features were then used to train various machine learning and deep learning models.

### 🤖 Models Implemented
Machine Learning Models
* Naïve Bayes
* Random Forest
* Voting Ensemble
* Extra Trees
* AdaBoost
* Stochastic Gradient Boosting
* Support Vector Machines (SVM)

### ⚙️ Implementation (via Phishector)
* A user-interactive Python program was developed with a menu system:
* Extract features from email folders.
* Choose between ML or DL classification.
* View performance of different models.

### 📊 Evaluation & Results
* Plotted evaluation metrics (e.g., accuracy) for each model across the datasets.

Neural Network significantly outperformed traditional ML models in terms of accuracy.

## 4) https://github.com/arvind-rs/phishing_detector/tree/master/Engineering%20Module

## 5) https://github.com/KartikTalwar/gmail.js
