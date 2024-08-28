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


## MLP Architecture

A Multi-Layer Perceptron (MLP) is a type of artificial neural network with multiple layers of interconnected nodes (neurons) that is widely used for various machine-learning tasks like classification and regression. Here's a breakdown of the MLP's structure and functioning:

### Structure
#### Input Layer:
- This layer consists of neurons that receive the features from the dataset directly. Each feature in the dataset corresponds to one neuron in the input layer. Thus, the number of neurons in this layer is equal to the number of features in the dataset.

#### Hidden Layers:
- These are layers situated between the input layer and the output layer. An MLP can have one or more hidden layers. The number of neurons in each hidden layer is a hyperparameter that can be chosen based on the complexity of the patterns in the data. These layers are crucial for detecting complex patterns.

#### Output Layer:
- This layer produces the final output or prediction based on the processed information from the hidden layers. The number of neurons in the output layer depends on the nature of the task:

  + For binary classification, there is typically one neuron that outputs a probability score.
  + For multi-class classification, there are as many neurons as there are classes, each producing a probability score for one class.
  + For regression tasks, there is usually one neuron that outputs the predicted continuous value.
![mlp](https://github.com/user-attachments/assets/1450d581-0e69-472c-aa07-cbae01be048f)

### Functioning
#### Initialization:
- The weights and biases of all neurons in the network are initially set. These parameters are usually initialized with small random values.

#### Forward Propagation:
- During training, the input data is fed through the network. Each neuron in a layer receives a weighted sum of inputs from the previous layer, applies an activation 
function to it, and passes the result to the next layer. Activation functions introduce non-linearity, allowing the model to learn complex relationships.

#### Loss Calculation:
The output of the network is compared to the actual target values to calculate the loss (error). Common loss functions include Mean Squared Error (MSE) for regression and 
Cross-Entropy for classification.

#### Backpropagation:
To minimize the loss, the network adjusts its weights and biases. The backpropagation algorithm computes the gradients of the loss concerning each parameter in the network. 
These gradients are used to update the weights and biases through optimization methods like Gradient Descent.

#### Training:
- The process of forward propagation, loss calculation, and backpropagation is repeated over many iterations (epochs) until the model converges to an optimal solution. 
Hyperparameters such as learning rate and the number of iterations can be adjusted.

#### Prediction:
Once trained, the MLP can make predictions on new, unseen data using the learned weights and biases through forward propagation.
