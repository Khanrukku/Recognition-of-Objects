# CIFAR-10 Image Classification using Convolutional Neural Networks

This project demonstrates how to build a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the CIFAR-10 dataset.

## Overview

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The goal is to classify these images into one of the following categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- Matplotlib
- NumPy

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your_username/cifar10-image-classification.git
cd cifar10-image-classification
pip install -r requirements.txt
```

## Dataset

The CIFAR-10 dataset is automatically downloaded by Keras when running the script. It is divided into training and testing sets.

## Code Structure

- `cifar10_image_classification.py`: Main script containing the model definition, training, and evaluation.
- `README.md`: This file, providing an overview of the project.

## Usage

Run the script to train the model and evaluate its performance:

```bash
python cifar10_image_classification.py
```

The script will train the CNN on the training data, validate it using the testing data, and display the accuracy metrics. It also generates predictions for a batch of test images and visualizes the results.

## Model Architecture

- Input Layer: 32x32 RGB images
- Convolutional Layers: Three sets of Conv2D + MaxPooling2D layers with ReLU activation
- Dropout Layers: Regularization to prevent overfitting
- Dense Layers: Fully connected layers with ReLU activation
- Output Layer: Softmax activation for 10 classes

## Results

The model achieves an above average accuracy.

## Example Predictions

The script generates predictions for a batch of test images and visualizes the results with the predicted and actual labels.

