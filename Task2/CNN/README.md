# MNIST Digit Recognition

A Convolutional Neural Network (CNN) to classify handwritten digits (0-9) from the MNIST dataset.

## 📊 Project Overview

This project implements a deep learning solution for recognizing handwritten digits using:
- **Dataset**: MNIST (60,000 training + 10,000 test images)
- **Model**: Convolutional Neural Network with 3 conv blocks
- **Framework**: TensorFlow/Keras
- **Accuracy**: >99% on test set

## 🚀 Features

- **Deep CNN Architecture**: 3 convolutional blocks with batch normalization and dropout
- **Data Preprocessing**: Normalization, reshaping, and one-hot encoding
- **Training Callbacks**: Early stopping, learning rate reduction, model checkpointing
- **Comprehensive Evaluation**: Confusion matrix, per-class accuracy, classification report
- **Visualizations**: Training curves, predictions, sample images
- **Model Persistence**: Saved models for deployment
- **Prediction Script**: Easy inference on new images

## 📁 Project Structure

```
Task2/
├── mnist_images/
│   ├── train/          # 60,000 training images
│   └── test/           # 10,000 test images
├── mnist_digit_recognition.py  # Main training script
├── predict_digit.py            # Prediction/inference script
├── load_data.py                # Data extraction script
├── README.md                   # This file
│
└── Generated Files:
    ├── mnist_cnn_model.h5      # Trained model (H5 format)
    ├── mnist_cnn_model.keras   # Trained model (Keras format)
    ├── best_mnist_model.h5     # Best checkpoint
    ├── training_history.pkl     # Training metrics
    └── Visualizations:
        ├── class_distribution_mnist.png
        ├── sample_digits.png
        ├── training_history.png
        ├── confusion_matrix_mnist.png
        ├── correct_predictions.png
        └── incorrect_predictions.png
```

## 🔧 Installation

```bash
# Install required packages
pip install tensorflow numpy matplotlib seaborn pillow scikit-learn
```

## 🎯 Usage

### Training the Model

Run the main training script:

```bash
cd Task2
python mnist_digit_recognition.py
```

This will:
1. Load 60,000 training and 10,000 test images
2. Preprocess the data (normalize, reshape, one-hot encode)
3. Build a CNN with 3 convolutional blocks
4. Train for up to 30 epochs with early stopping
5. Evaluate on test set
6. Generate visualizations and save the model

### Making Predictions

Use the prediction script:

```bash
python predict_digit.py
```

## 🏗️ Model Architecture

```
Input: 28x28x1 (grayscale image)
↓
Conv Block 1: Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
↓
Conv Block 2: Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
↓
Conv Block 3: Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
↓
Flatten
↓
Dense(256) → BatchNorm → Dropout(0.5)
↓
Dense(128) → BatchNorm → Dropout(0.5)
↓
Dense(10, softmax)
↓
Output: 10 classes (digits 0-9)
```

**Total Parameters**: ~1.5 million

## 📈 Model Performance

### Test Set Results:
- **Accuracy**: 99.53%
- **Loss**: 0.0144

### Per-Class Accuracy:
| Digit | Accuracy |
|-------|----------|
|   0   |  99.90%  |
|   1   |  99.56%  |
|   2   |  99.61%  |
|   3   |  99.90%  |
|   4   |  99.80%  |
|   5   |  99.33%  |
|   6   |  99.16%  |
|   7   |  99.42%  |
|   8   |  99.49%  |
|   9   |  99.11%  |

### Training Details:
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 128
- **Epochs**: ~15 (with early stopping)
- **Validation Split**: 10% of training data

## 🔍 Data Preprocessing

1. **Loading**: Images loaded from PNG files (28x28 grayscale)
2. **Normalization**: Pixel values scaled to [0, 1]
3. **Reshaping**: Added channel dimension (28, 28, 1)
4. **Label Encoding**: One-hot encoded for 10 classes

## 📊 Visualizations

The training script generates several visualizations:

1. **Class Distribution**: Shows balanced dataset across all digits
2. **Sample Images**: Grid of 40 sample digits with labels
3. **Training History**: Accuracy and loss curves over epochs
4. **Confusion Matrix**: Detailed per-class performance
5. **Predictions**: Examples of correct and incorrect classifications

## 🎓 Key Techniques Used

- **Convolutional Layers**: Extract spatial features from images
- **Batch Normalization**: Stabilize and speed up training
- **Dropout**: Prevent overfitting
- **Max Pooling**: Reduce spatial dimensions
- **Data Augmentation**: (Optional - not needed due to high baseline)
- **Early Stopping**: Prevent overtraining
- **Learning Rate Scheduling**: Adaptive learning rate reduction

## 📝 Training Callbacks

- **EarlyStopping**: Stop when validation loss stops improving (patience=5)
- **ReduceLROnPlateau**: Reduce learning rate when stuck (factor=0.5, patience=3)
- **ModelCheckpoint**: Save best model based on validation accuracy

## 🎯 Confusion Matrix Insights

The confusion matrix shows:
- **Diagonal dominance**: Most predictions are correct
- **Common confusions**: 
  - 4 ↔ 9 (similar curved shapes)
  - 3 ↔ 8 (both have curves)
  - 7 ↔ 1 (straight lines)

## 💡 Technical Details

### Image Format:
- **Size**: 28x28 pixels
- **Color**: Grayscale (1 channel)
- **Format**: PNG files
- **Naming**: `{label}_{index}.png`

### Model Optimization:
- **Kernel Size**: 3x3 (standard for CNNs)
- **Activation**: ReLU (fast and effective)
- **Output Activation**: Softmax (multi-class probability)
- **Regularization**: Dropout (0.25 for conv, 0.5 for dense)

## 📚 References

- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/)
- [CNN Architecture Best Practices](https://cs231n.github.io/)

## 👤 Author

Huzaifa Khalid

## ⭐ Acknowledgments

- Yann LeCun for the MNIST dataset
- TensorFlow team for the excellent framework
- Open source community for tools and libraries