# Building AI From Scratch: Regression, Classification, and Neural Networks in Python 

## Description
This repository contains implementations of linear and logistic regression algorithms, as well as complete neural networks, all coded from scratch in Python using only NumPy.

The regression models were originally developed while following Andrew Ng’s Machine Learning Specialization course on Coursera. These include core machine learning techniques such as gradient descent, cost function optimization, and basic data visualization.

Building on that foundation, this project now also features full neural network frameworks capable of both binary and multiclass classification. It supports:

- Forward and backward propagation
- Sigmoid, ReLU, and Softmax activations
- Binary and categorical cross-entropy loss
- He initialization
- Batch training
- Manual architecture configuration (hidden layers and units)

**No machine learning libraries (e.g., TensorFlow, PyTorch, scikit-learn) are used**. This project demonstrates a strong grasp of the core mechanics behind neural networks, including how training, activation functions, and backpropagation work under the hood.

## Contents
The contents of the repository are split into three directories:

- [linear-logistic-from-scratch](linear-logistic-from-scratch/):
  - [LinearandLogisticRegressionFromScratch.ipynb](linear-logistic-from-scratch/LinearandLogisticRegressionFromScratch.ipynb)
  - [LinearandLogisticRegressionFromScratchUtils.py](linear-logistic-from-scratch/LinearandLogisticRegressionFromScratchUtils.py) - utils only

- [nn-from-scratch](nn-from-scratch/):
  - [MNIST-DIGITS-NNBLUEPRINT.pdf](nn-from-scratch/MNIST-DIGITS-NNBLUEPRINT.pdf) - dense NN blueprint for MNIST digit classification
  - [NeuralNetworksFromScratch.ipynb](nn-from-scratch/NeuralNetworksFromScratch.ipynb)
  - [NeuralNetworksFromScratchUtils.py](nn-from-scratch/NeuralNetworksFromScratchUtils.py) - utils only
 
- [real-world-applications](real-world-applications)
  - [HeartDiseaseClassification.ipynb](real-world-applications/HeartDiseaseClassification.ipynb) - dense NN used for heart disease classification

## Notes
The multiclass dense NN is built entirely from scratch with NumPy. While it is fully functional, training can be slow on larger datasets compared to optimized libraries like TensorFlow or PyTorch. This design choice was intentional to reinforce core learning.

## How to Run
The `.ipynb` files (Jupyter Notebooks) can be run from top to bottom where they exhibit my machine learning journey. 

The `.py` files contain the reusable functions for linear/logistic regression models. One can import these functions for use in other projects or one can use them as standalone scripts.

## What One Will Learn
This project helped me understand the fundamentals of machine learning, including gradient descent, cost functions, and model training (supervised learning). 

## Author
James Milgram
