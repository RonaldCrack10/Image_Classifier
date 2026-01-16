## 🐾 Cat vs. Dog Classifier using CNN
This project implements a Convolutional Neural Network (CNN) to classify images of cats and dogs with high accuracy. Built using Python and Pytorch, it demonstrates the power of deep learning for computer vision tasks.

## 📌 Project Overview
The goal is to distinguish between domestic pets in images. The model learns spatial hierarchies of features, starting from simple edges to complex shapes like ears, whiskers, and tails.

Framework: Pytorch

Architecture: CNN

Dataset: to be seen in the repository

Optimizer: Stochastic Gradient Descend

## 🧠 Model Architecture
The network follows a classic sequential structure designed to extract features and then classify them:

Convolutional Layers: To extract spatial features from the input images.

Pooling Layers (Max Pooling): To reduce dimensionality and computational load.

Dense Layers: Fully connected layers to perform the final classification.

## 📂 Project Structure
├── data_preparation/
│   └── data_preparation.py   # Handles image resizing, normalization, and splitting
├── Model/
│   └── model.py              # Defines the CNN layers and architecture
├── model_training/
│   └── training.py           # Logic for fitting the model and saving weights
├── Tensorboard/
│   └── tensorboard.py        # Utilities for logging and visualization
├── model_testing/
│   └── testing.py            # Evaluates model performance on unseen data
├── requirements.txt          # Project dependencies
└── README.md

## 🚀 Getting Started
1. Prerequisites
Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

2. Installation
Bash

git clone https://github.com/your-username/image_classifier.git
cd cat-dog-cnn
pip install -r requirements.txt
3. Training the Model
To start the training process, run:

Bash

python model_training/training.py
## 📊 Results
The model achieved the following performance after 10 epochs:

Metric	Training	Validation
Accuracy	
Loss	

In Google Sheets exportieren

Performance Curves
Below are the training history plots showing how the model learned over time.

## 🔮 Future Improvements
Implement Transfer Learning using MobileNetV2 or EfficientNet for better accuracy.

Deploy the model as a web app using Streamlit or Flask.

Add Data Augmentation techniques (rotation, zoom, flip) to increase robustness.

## 📄 License
This project is licensed under the Apache License - see the LICENSE file for details.

Would you like me to generate the requirements.txt file for this project or help you write the specific Python code for the CNN layers?
