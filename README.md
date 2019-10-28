# Multi-layer Neural Networks for MNIST Dataset

We implement multi-layer neural network to classify the MNIST dataset in Python.

## Required Packages
* NumPy
* Pandas
* Matplotlib
* Pickle

## Dataset
We use the MNIST dataset for this neural network

## Activation
The activation class is used to apply activation function to layer output. 
The softmax activation function for the output layer is determined. We have

* sigmoid,
* ReLU,
* tanh

to be chosen from as our activation functions for hidden layers.

## Layer
The layers class keeps track of the input, weights, bias and their gradients of each layer.

## Neuralnet
The multi-layer neuralnet implemented with forward_pass and backward_pass with backpropagation.

## Main
Unchanged from starter file. 
