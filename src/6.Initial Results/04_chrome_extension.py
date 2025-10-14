# Chrome Extension Integration
# Model deployment code for Chrome Extension

import torch
import json
import pickle
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np
import re
from bs4 import BeautifulSoup

class PhishingDetector:
    """
    Phishing detection class for Chrome Extension integration
    """
    
    def __init__(self, model_path='./phishing_bert_model'):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        
    def load_model(self):
        """
        Load the trained BERT model
        """
        try:
            print("Loading phishing detection model...")
            
            # Check for GPU availability
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")
            if torch.cuda.is_available():
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
            self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
            
            # Move model to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.to(device)
                print("Model moved to GPU")
            
            self.model.eval()  # Set to evaluation mode
            self.is_loaded = True
            print("SUCCESS: Model loaded successfully!")
            return True
        except Exception as e:
            print(f"ERROR: Error loading model: {str(e)}")
            return False
    
    def preprocess_email(self, email_text):
        """
        Preprocess email text for prediction
        """
        if not email_text:
            return ""
        
        # Convert to string
        text = str(email_text)
        
        # Remove HTML tags
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        
        # Remove email headers
        text = re.sub(r'From:.*?\n', '', text, flags=re.IGNORECASE)
        text = re.sub(r'To:.*?\n', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Subject:.*?\n', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Date:.*?\n', '', text, flags=re.IGNORECASE)
        
        # Replace URLs with placeholder
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
        
        # Replace email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def predict_phishing(self, email_text, threshold=0.5):
        """
        Predict if email is phishing or legitimate
        
        Args:
            email_text (str): Email content to analyze
            threshold (float): Confidence threshold for classification
            
        Returns:
            dict: Prediction results
        """
        if not self.is_loaded:
            if not self.load_model():
                return {"error": "Model could not be loaded"}
        
        try:
            # Preprocess email
            processed_text = self.preprocess_email(email_text)
            
            if not processed_text:
                return {"error": "Empty email content"}
            
            # Tokenize
            inputs = self.tokenizer(
                processed_text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                phishing_prob = probabilities[0][1].item()
                legitimate_prob = probabilities[0][0].item()
                
                # Determine prediction
                is_phishing = phishing_prob > threshold
                confidence = max(phishing_prob, legitimate_prob)
                
                # Risk level assessment
                if phishing_prob > 0.8:
                    risk_level = "HIGH"
                elif phishing_prob > 0.6:
                    risk_level = "MEDIUM"
                elif phishing_prob > 0.4:
                    risk_level = "LOW"
                else:
                    risk_level = "VERY LOW"
            
            return {
                "is_phishing": bool(is_phishing),
                "phishing_probability": float(phishing_prob),
                "legitimate_probability": float(legitimate_prob),
                "confidence": float(confidence),
                "risk_level": risk_level,
                "threshold_used": threshold,
                "processed_text_length": len(processed_text)
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def batch_predict(self, email_list, threshold=0.5):
        """
        Predict phishing for multiple emails
        
        Args:
            email_list (list): List of email texts
            threshold (float): Confidence threshold
            
        Returns:
            list: List of prediction results
        """
        results = []
        for email_text in email_list:
            result = self.predict_phishing(email_text, threshold)
            results.append(result)
        return results
    
    def get_model_info(self):
        """
        Get model information
        """
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        
        return {
            "model_name": "DistilBERT Phishing Detector",
            "model_path": self.model_path,
            "max_length": 512,
            "num_labels": 2,
            "is_loaded": self.is_loaded
        }

def create_chrome_extension_files():
    """
    Create files needed for Chrome Extension integration
    """
    print("Creating Chrome Extension files...")
    
    # Manifest file
    manifest = {
        "manifest_version": 3,
        "name": "Phishing Email Detector",
        "version": "1.0",
        "description": "AI-powered phishing email detection using BERT",
        "permissions": [
            "activeTab",
            "storage"
        ],
        "action": {
            "default_popup": "popup.html",
            "default_title": "Phishing Detector"
        },
        "content_scripts": [
            {
                "matches": ["<all_urls>"],
                "js": ["content.js"]
            }
        ],
        "background": {
            "service_worker": "background.js"
        },
        "web_accessible_resources": [
            {
                "resources": ["model/*"],
                "matches": ["<all_urls>"]
            }
        ]
    }
    
    with open('chrome_extension/manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Popup HTML
    popup_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {
                width: 350px;
                padding: 20px;
                font-family: Arial, sans-serif;
            }
            .header {
                text-align: center;
                margin-bottom: 20px;
            }
            .detection-area {
                margin: 20px 0;
            }
            textarea {
                width: 100%;
                height: 100px;
                margin: 10px 0;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            button {
                width: 100%;
                padding: 10px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            button:hover {
                background-color: #45a049;
            }
            .result {
                margin-top: 20px;
                padding: 15px;
                border-radius: 4px;
                font-weight: bold;
            }
            .phishing {
                background-color: #ffebee;
                color: #c62828;
                border: 1px solid #ffcdd2;
            }
            .legitimate {
                background-color: #e8f5e8;
                color: #2e7d32;
                border: 1px solid #c8e6c9;
            }
            .risk-high { color: #d32f2f; }
            .risk-medium { color: #f57c00; }
            .risk-low { color: #388e3c; }
            .risk-very-low { color: #1976d2; }
        </style>
    </head>
    <body>
        <div class="header">
            <h2>🛡️ Phishing Detector</h2>
            <p>AI-powered email analysis</p>
        </div>
        
        <div class="detection-area">
            <label for="email-text">Email Content:</label>
            <textarea id="email-text" placeholder="Paste email content here..."></textarea>
            <button id="analyze-btn">Analyze Email</button>
        </div>
        
        <div id="result" class="result" style="display: none;"></div>
        
        <script src="popup.js"></script>
    </body>
    </html>
    """
    
    with open('chrome_extension/popup.html', 'w') as f:
        f.write(popup_html)
    
    # Popup JavaScript
    popup_js = """
    document.addEventListener('DOMContentLoaded', function() {
        const analyzeBtn = document.getElementById('analyze-btn');
        const emailText = document.getElementById('email-text');
        const resultDiv = document.getElementById('result');
        
        analyzeBtn.addEventListener('click', function() {
            const text = emailText.value.trim();
            
            if (!text) {
                alert('Please enter email content to analyze.');
                return;
            }
            
            // Send message to background script
            chrome.runtime.sendMessage({
                action: 'analyze_email',
                emailText: text
            }, function(response) {
                displayResult(response);
            });
        });
        
        function displayResult(result) {
            if (result.error) {
                resultDiv.innerHTML = `<div style="color: red;">Error: ${result.error}</div>`;
            } else {
                const isPhishing = result.is_phishing;
                const riskLevel = result.risk_level.toLowerCase().replace(' ', '-');
                const confidence = (result.confidence * 100).toFixed(1);
                
                resultDiv.className = `result ${isPhishing ? 'phishing' : 'legitimate'}`;
                resultDiv.innerHTML = `
                    <div>
                        <strong>${isPhishing ? '🚨 PHISHING DETECTED' : '✅ Legitimate Email'}</strong>
                    </div>
                    <div style="margin-top: 10px;">
                        <div>Risk Level: <span class="risk-${riskLevel}">${result.risk_level}</span></div>
                        <div>Confidence: ${confidence}%</div>
                        <div>Phishing Probability: ${(result.phishing_probability * 100).toFixed(1)}%</div>
                    </div>
                `;
            }
            
            resultDiv.style.display = 'block';
        }
    });
    """
    
    with open('chrome_extension/popup.js', 'w') as f:
        f.write(popup_js)
    
    # Background script
    background_js = """
    // Background script for Chrome Extension
    chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
        if (request.action === 'analyze_email') {
            // In a real implementation, this would call the Python model
            // For now, we'll simulate the response
            simulatePhishingDetection(request.emailText, sendResponse);
            return true; // Keep message channel open for async response
        }
    });
    
    function simulatePhishingDetection(emailText, callback) {
        // Simulate AI analysis
        setTimeout(() => {
            const phishingKeywords = ['urgent', 'click here', 'verify account', 'suspended', 'immediately'];
            const text = emailText.toLowerCase();
            
            let phishingScore = 0;
            phishingKeywords.forEach(keyword => {
                if (text.includes(keyword)) {
                    phishingScore += 0.2;
                }
            });
            
            // Add random factor for simulation
            phishingScore += Math.random() * 0.3;
            phishingScore = Math.min(phishingScore, 1);
            
            const result = {
                is_phishing: phishingScore > 0.5,
                phishing_probability: phishingScore,
                legitimate_probability: 1 - phishingScore,
                confidence: Math.max(phishingScore, 1 - phishingScore),
                risk_level: phishingScore > 0.8 ? 'HIGH' : 
                           phishingScore > 0.6 ? 'MEDIUM' : 
                           phishingScore > 0.4 ? 'LOW' : 'VERY LOW',
                threshold_used: 0.5
            };
            
            callback(result);
        }, 1000);
    }
    """
    
    with open('chrome_extension/background.js', 'w') as f:
        f.write(background_js)
    
    print("SUCCESS: Chrome Extension files created!")

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = PhishingDetector()
    
    # Test with sample emails
    sample_emails = [
        "Dear Customer, Your account has been suspended. Click here to verify: http://fake-bank.com/verify",
        "Hi John, Thanks for the meeting yesterday. Let's schedule a follow-up next week.",
        "URGENT: Your PayPal account will be closed in 24 hours unless you verify your identity immediately!"
    ]
    
    print("Testing Phishing Detector...")
    
    for i, email in enumerate(sample_emails, 1):
        print(f"\nTest Email {i}:")
        print(f"Content: {email[:100]}...")
        
        result = detector.predict_phishing(email)
        
        if 'error' in result:
            print(f"ERROR: Error: {result['error']}")
        else:
            print(f"Prediction: {'PHISHING' if result['is_phishing'] else 'Legitimate'}")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Phishing Probability: {result['phishing_probability']:.2f}")
    
    # Create Chrome Extension files
    create_chrome_extension_files()
    
    print("\nPhishing Detector Ready for Chrome Extension!")
