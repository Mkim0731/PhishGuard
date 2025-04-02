🧱 Goal

Deploy the trained phishing detection model and integrate it into the PhishGuard Chrome Extension using a lightweight backend API. The goal is to enable real-time email analysis in Gmail through content script injection and model inference via HTTP requests.

Key Objectives:

Expose model through a local Flask/FastAPI server

Handle email prediction requests from extension background script

Display phishing warnings to the user based on prediction result

Ensure fast inference and minimal performance overhead

🚪 Project Structure

phishguard-extension/
├── manifest.json
├── background.js
├── content.js
├── popup.html
├── popup.js
├── styles.css
├── icons/
├── api/
│   └── model_api.py     # Flask/FastAPI app
└── model/
    └── phishing_model.pkl

📂 manifest.json – Define Extension Behavior

{
  "manifest_version": 3,
  "name": "PhishGuard",
  "version": "1.0",
  "description": "Detect phishing emails using ML",
  "permissions": ["scripting", "activeTab", "storage"],
  "host_permissions": ["https://mail.google.com/"],
  "background": {
    "service_worker": "background.js"
  },
  "action": {
    "default_popup": "popup.html",
    "default_icon": {
      "16": "icons/icon16.png",
      "48": "icons/icon48.png",
      "128": "icons/icon128.png"
    }
  },
  "content_scripts": [
    {
      "matches": ["https://mail.google.com/*"],
      "js": ["content.js"]
    }
  ]
}

🤖 content.js – Extract Email Text from Gmail

setTimeout(() => {
  const emailBody = document.querySelector('.ii.gt');
  const emailText = emailBody?.innerText || '';
  chrome.runtime.sendMessage({ emailText });
}, 3000);

🛠️ background.js – Send Data to Model API

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  fetch('http://localhost:5000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ email: request.emailText })
  })
    .then(res => res.json())
    .then(data => {
      if (data.prediction === 1) {
        alert("⚠️ This email might be a phishing attempt!");
      }
    });
});

📑 model_api.py – Inference Server (Flask)

from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open("model/phishing_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    email = request.json['email']
    vector = vectorizer.transform([email])
    prediction = model.predict(vector)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(port=5000)

Dependencies: pip install flask scikit-learn

🔍 Deployment Notes

Localhost is used for development; deploy via ngrok or cloud when going live

Consider converting model to ONNX or TensorFlow Lite for JS-based inference in future

Secure the endpoint with token-based validation before production release

🚀 Future Enhancements

Replace Flask with FastAPI for async performance and auto docs

Add confidence scores and explanations using LIME or SHAP

Implement background caching to avoid duplicate predictions

Integrate with Gmail.js for deeper email context and DOM interaction

Explore full in-browser inference using TensorFlow.js or ONNX.js

🌟 Conclusion

This deployment and integration phase successfully bridges the trained phishing model with a real-world Chrome extension using a lightweight Flask API. It enables PhishGuard to scan Gmail inboxes for phishing attempts in real-time, enhancing user security with minimal friction.

