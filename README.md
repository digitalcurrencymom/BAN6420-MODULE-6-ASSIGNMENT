# BAN6420-MODULE-6-ASSIGNMENT
FASHION MNIST CNN PROJECT

## Overview
This project implements a Convolutional Neural Network (CNN) using both Python and R to classify images from the Fashion MNIST dataset. It serves as a foundational model for future applications in profile image classification and targeted marketing. The CNN architecture includes six layers and is designed for reproducibility, portability, and educational clarity.

## Objectives
Build a 6-layer CNN using Keras in Python and R

Train on the Fashion MNIST dataset

Predict labels for at least two test images

Generate visual output and classification report

Ensure compatibility with local and cloud environments (e.g., Google Colab)

Provide fallback logic for environments without TensorFlow

## Technologies Used
Language	Frameworks/Libraries
Python	TensorFlow, Keras, NumPy, Matplotlib, scikit-learn
R	Keras (R interface), TensorFlow
CI/CD	GitHub Actions (Python 3.10)
Testing	Pytest, pytest-console-scripts

## Repository Structure
Code
Fashion_MNIST_CNN/

fashion_cnn.py               # Python CNN script with fallback logic
fashion_cnn.R                # R script using keras for R
predictions_output.txt       # Predicted labels for two test images
fashion_mnist_plot.png       # Visualization of predictions
 requirements.txt             # Python dependencies
requirements_extra.txt       # Extended dependencies for testing
CONTRIBUTING.md              # Contribution guidelines
README.md                    # Project documentation
github/workflows/ci.yml     # GitHub Actions workflow for CI

## How to Run
 Python Instructions
Option A: Using Conda (Recommended)
bash
conda create -n tf310 python=3.10 pip -y
conda activate tf310
pip install -r requirements.txt
python fashion_cnn.py
Option B: Using Virtualenv
bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python fashion_cnn.py

## Outputs
predictions_output.txt: Contains predicted labels for two test images

fashion_mnist_plot.png: Side-by-side visualization of predictions
## R Instructions
Install the keras R package and configure TensorFlow backend:

r
install.packages("keras")
library(keras)
install_keras()
Run the script:

r
Rscript fashion_cnn.R

## Testing & CI
Run Tests Locally
bash
pip install -r requirements_extra.txt
pytest -q
GitHub Actions
CI workflow runs on Python 3.10

Located at .github/workflows/ci.yml

## Notes
If TensorFlow is not available, fashion_cnn.py falls back to training a simple MLP using scikit-learn on the digits dataset.

For cloud execution, use the included Colab notebook (not shown here) and document provenance in run_notes.txt.

## Educational Value
This project demonstrates:

Cross-language implementation (Python + R)

Model portability and fallback logic

Reproducible training and prediction

Integration of testing and CI/CD

Clear documentation and modular design

 ## Author
Kendra Onah 
Junior Machine Learning Researcher 
Nexford University
