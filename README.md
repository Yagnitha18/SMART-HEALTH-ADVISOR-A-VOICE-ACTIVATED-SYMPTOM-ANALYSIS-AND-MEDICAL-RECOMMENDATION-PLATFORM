# SMART HEALTH ADVISOR: A VOICE-ACTIVATED SYMPTOM ANALYSIS AND MEDICAL RECOMMENDATION PLATFORM

## Abstract

Many people suffer from various diseases but lack access to proper medical care. To address  this issue, we introduced a FLASK app with a voice-enabled symptom input feature that provides medication recommendations. Users can report symptoms through typing or speech, leveraging a custom Multi-Layer Perceptron (MLP) classifier model to predict diseases. The app utilizes the Web Speech API for voice recognition and spaCy for natural language processing, ensuring accurate symptom extraction. The system predicts the most likely disease and provides comprehensive information, including disease descriptions, recommended precautions, medications, dietary advice, and workout suggestions, improving accessibility and user experience. Our study evaluates the performance of several machine learning algorithms on a synthetic classification dataset. We assessed Support Vector Classifier (SVC), Random Forest, Gradient Boosting, and K-Nearest Neighbors (KNN) algorithms for accuracy, confusion matrix, classification report, and cross-validation scores. KNN exhibited the accuracy at 84%, followed by Random Forest at 79%, Gradient Boosting at 77.5%, and SVC at 70%. The custom MLP model demonstrated superior performance on a different dataset with 100% accuracy, precision, recall, and F1 score. The findings suggest that while traditional models are effective, neural network-based approaches like MLP achieve exceptional performance, particularly in complex datasets.

Key Words: Multi-Layer Perceptron (MLP) classifier, Flask app, Web Speech API, spaCy


## Objective

- To develop a Flask-based web application for symptom-based disease prediction and medical recommendations.
- To integrate a voice recognition feature for symptom input, enhancing user accessibility.
- To implement a custom Multi-Layer Perceptron (MLP) classifier for accurate disease prediction based on user-reported symptoms.
- To provide users comprehensive information on predicted diseases, including descriptions, precautions, medications, dietary advice, and workout suggestions.
- To improve the overall user experience by making the system interactive and user-friendly.


