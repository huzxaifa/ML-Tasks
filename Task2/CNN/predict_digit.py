"""
Load trained model and predict digits from images
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras

print("="*80)
print(" "*25 + "MNIST DIGIT PREDICTOR")
print("="*80)

def load_model():
    """Load the trained CNN model"""

    model_files = ['mnist_cnn.h5', 'mnist_cnn.keras']
    
    for model_file in model_files:
        try:
            model = keras.models.load_model(model_file)
            print(f"\n[OK] Model loaded successfully from: {model_file}")
            return model
        except:
            continue
    
    print("\n[ERROR] Could not load model. Please train the model first.")
    print("Expected files: best_mnist_cnn.h5, mnist_cnn_final.h5, or mnist_cnn_model.h5")
    return None

def preprocess_image(image_path):
    """
    Preprocess an image for prediction
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed image ready for model
    """
    # Load image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    
    # Convert to array and normalize
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    
    # Reshape for CNN
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array, img

def predict_digit(model, image_path, display=True):
    """
    Predict digit from an image
    
    Args:
        model: Trained Keras model
        image_path: Path to image file
        display: Whether to display the image and prediction
        
    Returns:
        Dictionary with prediction results
    """
    # Preprocess image
    img_processed, img_original = preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(img_processed, verbose=0)[0]
    predicted_digit = np.argmax(predictions)
    confidence = predictions[predicted_digit]
    
    # Display results
    if display:
        plt.figure(figsize=(10, 4))
        
        # Show image
        plt.subplot(1, 2, 1)
        plt.imshow(img_original, cmap='gray')
        plt.title(f'Input Image', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Show predictions
        plt.subplot(1, 2, 2)
        plt.bar(range(10), predictions, color='steelblue', alpha=0.7)
        plt.axhline(y=confidence, color='r', linestyle='--', linewidth=2, label=f'Max: {confidence:.2%}')
        plt.xlabel('Digit', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.title(f'Predicted: {predicted_digit} (Confidence: {confidence:.2%})', 
                 fontsize=14, fontweight='bold')
        plt.xticks(range(10))
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return {
        'predicted_digit': int(predicted_digit),
        'confidence': float(confidence),
        'probabilities': predictions.tolist()
    }

def predict_multiple_images(model, image_paths):
    """
    Predict digits from multiple images
    
    Args:
        model: Trained Keras model
        image_paths: List of image paths
    """
    n_images = len(image_paths)
    cols = min(5, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    results = []
    
    for i, img_path in enumerate(image_paths):
        # Preprocess and predict
        img_processed, img_original = preprocess_image(img_path)
        predictions = model.predict(img_processed, verbose=0)[0]
        predicted_digit = np.argmax(predictions)
        confidence = predictions[predicted_digit]
        
        results.append({
            'path': img_path,
            'predicted': int(predicted_digit),
            'confidence': float(confidence)
        })
        
        # Display
        axes[i].imshow(img_original, cmap='gray')
        axes[i].set_title(f'Pred: {predicted_digit} ({confidence:.1%})', fontsize=11)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Batch Digit Recognition', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return results

def main():
    """Main function to demonstrate digit prediction"""
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Example: Predict from test images
    print("\n" + "="*80)
    print("TESTING WITH SAMPLE IMAGES")
    print("="*80)
    
    import os
    
    # Get some test images
    test_dir = '../mnist_images/test'
    test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir)[:10]]
    
    print(f"\nTesting with {len(test_images)} sample images...")
    
    # Predict first image in detail
    print("\n" + "-"*80)
    print("Single Image Prediction:")
    print("-"*80)
    result = predict_digit(model, test_images[0], display=True)
    print(f"\nResult:")
    print(f"  Predicted Digit: {result['predicted_digit']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    
    # Predict multiple images
    print("\n" + "-"*80)
    print("Batch Prediction:")
    print("-"*80)
    results = predict_multiple_images(model, test_images)
    
    print(f"\nBatch Results:")
    for i, res in enumerate(results, 1):
        print(f"  Image {i}: Predicted = {res['predicted']}, Confidence = {res['confidence']:.2%}")
    
    # Calculate accuracy on sample
    actual_labels = [int(os.path.basename(path).split('_')[0]) for path in test_images]
    predicted_labels = [res['predicted'] for res in results]
    accuracy = sum([a == p for a, p in zip(actual_labels, predicted_labels)]) / len(actual_labels)
    
    print(f"\nSample Accuracy: {accuracy:.2%}")
    
    print("\n" + "="*80)
    print("Prediction complete!")
    print("="*80)

if __name__ == "__main__":
    main()


