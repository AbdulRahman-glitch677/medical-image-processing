# Medical Image Classification — Pneumonia Detection

## Overview
A deep learning model that classifies chest X-ray images to detect pneumonia using
a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

## Dataset
[Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

- Classes: NORMAL, PNEUMONIA
- Train: 5,216 images
- Validation: 16 images
- Test: 624 images

## Model Architecture
- 4x Convolutional blocks with Batch Normalization and MaxPooling
- Dense layer (512 units) with Dropout (0.5) for regularization
- Sigmoid output for binary classification
- Total parameters: 19.2M

## Training Details
- Optimizer: Adam
- Loss: Binary Crossentropy
- Image size: 224x224
- Batch size: 32
- Early stopping applied (stopped at epoch 7/10, best weights restored)

## Results

| Metric    | NORMAL | PNEUMONIA |
|-----------|--------|-----------|
| Precision | 0.61   | 0.89      |
| Recall    | 0.85   | 0.67      |
| F1-Score  | 0.71   | 0.77      |

Overall Accuracy: 74%

## Note on Model Weights
The trained model file exceeds GitHub's 100MB limit and is not included here.
To reproduce the model, run the notebook end-to-end on Kaggle with GPU enabled.

## How to Run
1. Open the notebook on Kaggle
2. Add the Chest X-Ray dataset
3. Enable GPU: Settings -> Accelerator -> GPU T4
4. Run all cells in order
5. Use the prediction cell to test on any chest X-ray image

## Tools
Python, TensorFlow, Keras, NumPy, Matplotlib, Seaborn, scikit-learn
