Task 3: Model Selection and Training – Research on Existing Technologies and Methodologies

🌟 Goal

The goal of this phase is to identify, train, and evaluate various machine learning models to accurately classify emails as phishing (1) or legitimate (0) using engineered features. This subtask is crucial to the PhishGuard Google Chrome extension, where accurate classification directly translates to real-time threat detection.

Key Objectives:

Apply supervised learning classification models

Evaluate training vs test performance to detect overfitting

Compare results across models and choose the best performer

Save the trained model for real-time use in the browser extension

🔧 Models and Tools Used

Model

Description

Train Accuracy

Test Accuracy

Notes

XGBoost

Gradient boosted trees; fast and accurate

0.866

0.864

Best performer; saved for use in production

Multilayer Perceptrons

Deep feedforward neural network

0.858

0.863

High-performing; slight risk of overfitting

Random Forest

Ensemble of decision trees with bagging

0.814

0.834

Good baseline; interpretable

Decision Tree

Simple interpretable tree structure

0.810

0.826

Fast and easy to debug; prone to overfitting

Autoencoder Neural Network

Learns compressed representation; unsupervised anomaly detection

0.819

0.818

Interesting for phishing reconstruction scoring

Support Vector Machine

Linear classifier with maximum margin

0.798

0.818

Decent accuracy; scalable with linear kernel

📊 Performance Analysis

Top performer: XGBoost showed the highest test accuracy (0.864) and generalization power, making it the final choice for deployment.

MLP had similar performance, but training time and sensitivity to hyperparameters made it slightly less favorable.

Autoencoder showed promise as an unsupervised alternative for anomaly scoring.

Decision Tree & Random Forest were excellent for feature importance analysis.

Feature Importance Example (from XGBoost & Random Forest)

URL length

Suspicious domain patterns

Header anomalies (SPF, DKIM)

Word-level features from email body

🔬 Experimental Setup

Train/Test Split: 80/20

Metrics: Accuracy, Feature Importance, Model Interpretability

Storage: Best model (XGBoost) saved using pickle for Chrome extension integration

# Saving the XGBoost model
import pickle
pickle.dump(xgb, open("XGBoostClassifier.pickle.dat", "wb"))

📈 Additional Insights from Literature

Multi-model fusion, like the MultiText-LP approach (Colhak et al., 2024), improves performance by combining MLPs with pre-trained NLP embeddings (e.g., RoBERTa, CANINE).

Ensemble learning is shown to be resilient to noise and imbalanced data — especially useful in phishing datasets where legitimate emails dominate.

Big Data scale: While not yet implemented, scalable training using Spark MLlib or H2O.ai could be beneficial as data volume grows.

🔍 Future Enhancements

Integrate transformer embeddings (e.g., BERT) with tabular features via fusion pipelines

Use semi-supervised learning on unlabeled email datasets to increase model robustness

Explore anomaly detection for zero-day phishing attacks

Evaluate F1-score and AUC for better understanding of classifier bias

🔹 Conclusion

Through rigorous experimentation and evaluation, XGBoost emerged as the best model for phishing detection in this project. The balance between performance, interpretability, and scalability makes it ideal for deployment in a real-time environment like a Chrome extension.

This model selection and training phase sets the foundation for the intelligent detection layer of PhishGuard.

