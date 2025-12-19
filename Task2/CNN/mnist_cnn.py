import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import pickle

# TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print("="*80)
print(" "*20 + "MNIST DIGIT RECOGNITION - CNN")
print("="*80)
print(f"\nTensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
print("\n[OK] All libraries imported successfully!\n")

# ============================================================================
# 1. LOAD DATASET
# ============================================================================

print("="*80)
print("LOADING MNIST DATASET")
print("="*80)

def load_images_from_folder(folder_path):
    """Load images and labels from folder"""
    images = []
    labels = []
    
    files = sorted(os.listdir(folder_path))
    print(f"Loading from: {folder_path}")
    print(f"Total files: {len(files)}")
    
    for i, filename in enumerate(files):
        if filename.endswith('.png'):
            label = int(filename.split('_')[0])
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('L')
            img_array = np.array(img)
            images.append(img_array)
            labels.append(label)
        
        if (i + 1) % 10000 == 0:
            print(f"  Loaded {i + 1} images...")
    
    return np.array(images), np.array(labels)

# Load data
print("\nLoading training data...")
X_train, y_train = load_images_from_folder('../mnist_images/train')
print(f"[OK] Training: {X_train.shape}")

print("\nLoading test data...")
X_test, y_test = load_images_from_folder('../mnist_images/test')
print(f"[OK] Test: {X_test.shape}")

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================

print("\n" + "="*80)
print("DATA PREPROCESSING")
print("="*80)

# Normalize
print("\n1. Normalizing pixel values...")
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape for CNN
print("2. Reshaping for CNN...")
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
print(f"   Training: {X_train.shape}")
print(f"   Test: {X_test.shape}")

# One-hot encode labels
print("3. One-hot encoding labels...")
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Create validation split
print("4. Creating validation set (10%)...")
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train_cat, test_size=0.1, random_state=42, stratify=y_train
)
print(f"   Training: {X_train_final.shape}")
print(f"   Validation: {X_val.shape}")

# ============================================================================
# 3. BUILD CNN MODEL
# ============================================================================

print("\n" + "="*80)
print("BUILDING CNN MODEL")
print("="*80)

model = models.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Dense layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n[OK] Model created!")
print("\nModel Summary:")
model.summary()

# ============================================================================
# 4. DATA AUGMENTATION (Optional)
# ============================================================================

print("\n" + "="*80)
print("DATA AUGMENTATION")
print("="*80)

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

datagen.fit(X_train_final)
print("[OK] Data augmentation configured")

# ============================================================================
# 5. TRAIN MODEL
# ============================================================================

print("\n" + "="*80)
print("TRAINING MODEL")
print("="*80)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ModelCheckpoint('best_mnist_cnn.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
]

print("\nTraining with data augmentation...")
print("="*80)

history = model.fit(
    datagen.flow(X_train_final, y_train_final, batch_size=128),
    epochs=30,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

print("\n[OK] Training completed!")

# ============================================================================
# 6. EVALUATE MODEL
# ============================================================================

print("\n" + "="*80)
print("MODEL EVALUATION")
print("="*80)

# Test set evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Predictions
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

# Per-class accuracy
print("\nPer-Class Accuracy:")
print("-"*50)
for digit in range(10):
    mask = y_test == digit
    digit_acc = accuracy_score(y_test[mask], y_pred[mask])
    print(f"  Digit {digit}: {digit_acc*100:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.title('Confusion Matrix - CNN', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_cnn.png', dpi=300)
print("\n[OK] Confusion matrix saved")
plt.show()

# Classification report
print("\n" + "="*80)
print("CLASSIFICATION REPORT")
print("="*80)
print(classification_report(y_test, y_pred, target_names=[f'Digit {i}' for i in range(10)]))

# Training history
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].plot(history.history['accuracy'], label='Training', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history['loss'], label='Training', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_cnn.png', dpi=300)
print("[OK] Training history saved")
plt.show()

# Visualize predictions
correct_idx = np.where(y_pred == y_test)[0]
incorrect_idx = np.where(y_pred != y_test)[0]

print(f"\nCorrect: {len(correct_idx)} ({len(correct_idx)/len(y_test)*100:.2f}%)")
print(f"Incorrect: {len(incorrect_idx)} ({len(incorrect_idx)/len(y_test)*100:.2f}%)")

# Show correct predictions
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Correct Predictions', fontsize=16, fontweight='bold')

for i, idx in enumerate(correct_idx[:10]):
    ax = axes[i // 5, i % 5]
    ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    ax.set_title(f'True: {y_test[idx]}, Pred: {y_pred[idx]}\nConf: {y_pred_proba[idx][y_pred[idx]]:.2%}',
                fontsize=10, color='green')
    ax.axis('off')

plt.tight_layout()
plt.savefig('correct_predictions_cnn.png', dpi=300)
print("[OK] Correct predictions saved")
plt.show()

# Show incorrect predictions
if len(incorrect_idx) > 0:
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Incorrect Predictions', fontsize=16, fontweight='bold')
    
    for i, idx in enumerate(incorrect_idx[:10]):
        ax = axes[i // 5, i % 5]
        ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        ax.set_title(f'True: {y_test[idx]}, Pred: {y_pred[idx]}\nConf: {y_pred_proba[idx][y_pred[idx]]:.2%}',
                    fontsize=10, color='red')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('incorrect_predictions_cnn.png', dpi=300)
    print("[OK] Incorrect predictions saved")
    plt.show()

# ============================================================================
# 7. SAVE MODEL
# ============================================================================

print("\n" + "="*80)
print("SAVING MODEL")
print("="*80)

model.save('mnist_cnn_final.h5')
print("\n[OK] Model saved as: mnist_cnn_final.h5")

model.save('mnist_cnn_final.keras')
print("[OK] Model saved as: mnist_cnn_final.keras")

# Save history
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
print("[OK] Training history saved")

# ============================================================================
# 8. FINAL SUMMARY
# ============================================================================

print("\n" + "#"*80)
print("#" + " "*78 + "#")
print("#" + " "*15 + "MNIST CNN DIGIT RECOGNITION - FINAL REPORT" + " "*21 + "#")
print("#" + " "*78 + "#")
print("#"*80)

print("\n1. DATASET:")
print("   " + "="*70)
print(f"   Training samples: 60,000")
print(f"   Test samples: 10,000")
print(f"   Image size: 28x28 pixels")
print(f"   Classes: 10 (digits 0-9)")

print("\n2. MODEL:")
print("   " + "="*70)
print(f"   Architecture: Convolutional Neural Network (CNN)")
print(f"   Total parameters: {model.count_params():,}")
print(f"   Optimizer: Adam")
print(f"   Loss: Categorical Crossentropy")

print("\n3. TRAINING:")
print("   " + "="*70)
print(f"   Epochs: {len(history.history['loss'])}")
print(f"   Data augmentation: Yes")
print(f"   Final training accuracy: {history.history['accuracy'][-1]*100:.2f}%")
print(f"   Final validation accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")

print("\n4. TEST PERFORMANCE:")
print("   " + "="*70)
print(f"   Test Accuracy: {test_accuracy*100:.2f}%")
print(f"   Test Loss: {test_loss:.4f}")
print(f"   Correct predictions: {len(correct_idx)}/{len(y_test)}")

print("\n5. FILES GENERATED:")
print("   " + "="*70)
print("   [OK] mnist_cnn_final.h5")
print("   [OK] mnist_cnn_final.keras")
print("   [OK] best_mnist_cnn.h5")
print("   [OK] training_history.pkl")
print("   [OK] confusion_matrix_cnn.png")
print("   [OK] training_history_cnn.png")
print("   [OK] correct_predictions_cnn.png")
print("   [OK] incorrect_predictions_cnn.png")

print("\n" + "#"*80)
print("#" + " "*25 + "PROJECT COMPLETED SUCCESSFULLY!" + " "*24 + "#")
print("#"*80 + "\n")

print(f"[OK] CNN Model achieved {test_accuracy*100:.2f}% accuracy!")
print("[OK] Model ready for deployment!")


