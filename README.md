# MNIST Handwritten Digit Classifier 
This repository contains a deep learning project that trains a neural network to recognize handwritten digits (0–9) using the MNIST dataset. The model is implemented in PyTorch and demonstrates the full machine learning pipeline, including data preprocessing, model design, training, evaluation, and visualization of predictions.

# Project Overview
The goal of this project is to build and train a digit classification model from scratch using PyTorch. The model takes 28×28 grayscale images of handwritten digits and predicts the correct digit label. After training, the model achieves approximately 97% accuracy on the test dataset.

# Technologies Used
Python
PyTorch
Torchvision
NumPy
Matplotlib
Jupyter Notebook

# Dataset
MNIST Handwritten Digits Dataset
Each image is 28×28 pixels, grayscale, labeled from 0 to 9.

# Model Architecture
The model is a fully connected neural network with:
- Input layer of 784 units (28×28 flattened image)
- One hidden layer with 128 neurons and ReLU activation
- Output layer with 10 neurons (one for each digit class)

# Loss Function and Optimizer
Loss Function: CrossEntropyLoss
Optimizer: Adam (learning rate = 0.001)

# Training
The model is trained for multiple epochs using mini-batches of data. During training, the model learns to minimize classification error by adjusting its weights through backpropagation.

# Evaluation
Model performance is evaluated on a separate test dataset. The final accuracy is approximately 97%. Example predictions are visualized alongside their true labels to qualitatively assess model behavior.

# Visualization
The repository includes code to display sample predictions in a grid format, showing both the predicted digit and the true label for each image.

How to Run
1) Install required dependencies (PyTorch, torchvision, matplotlib, numpy).
2) Open the Jupyter notebook.
3) Run all cells sequentially to train the model and view results.

# Future Improvements
- Replace the fully connected network with a convolutional neural network (CNN)
- Add confusion matrix visualization
- Save and reload trained models
- Experiment with different architectures and hyperparameters

Author
Saba
