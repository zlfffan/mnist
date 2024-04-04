# MNIST Classification

## Introduction
This repository contains code for training a fully Convolutional Network (FCN) model to classify handwritten digits using the MNIST dataset. This experiment adopts the SGD optimizer and cross-entropy loss function to train the network for 10 epochs, achieving a final accuracy of 99.6%.

## Installation
Clone this repository:
```
git clone https://github.com/zlfffan/mnist-classification.git
```

## Usage
1. Run `train.py` to train the CNN model.
2. Run `predict.py` to view the model's prediction results.
3. Run the following command to view the training process:
```
tensorboard --logdir=run
```