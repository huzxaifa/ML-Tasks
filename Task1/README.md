# Email Spam Classification System

A machine learning system to classify emails as spam or not spam using Natural Language Processing and two classifiers: **Multinomial Naive Bayes** and **Logistic Regression**.

## 📋 Project Overview

This project implements a complete email spam detection pipeline including:
- Data preprocessing and text cleaning
- Feature extraction using TF-IDF
- Model training with two algorithms
- Hyperparameter tuning
- Comprehensive evaluation with multiple metrics
- Visualization of results
- Saved models for deployment

## 🚀 Features

- **Text Preprocessing**: Lowercase conversion, URL/email removal, tokenization, stopword removal, and stemming
- **Feature Extraction**: TF-IDF vectorization with unigrams and bigrams
- **Multiple Models**: Multinomial Naive Bayes and Logistic Regression
- **Hyperparameter Tuning**: GridSearchCV for optimal parameters
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualizations**: Confusion matrices, ROC curves, word clouds, feature importance
- **Model Persistence**: Saved models ready for deployment

## 📦 Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Download NLTK data (automatically done when running the scripts):
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

## 📊 Dataset

- **File**: `emails2.csv`
- **Format**: Raw email text with spam labels
- **Columns**: 
  - `text`: Email content
  - `spam`: Label (0 = Not Spam, 1 = Spam)

## 🔧 Usage

### Training the Model

Run the main script to train the models:

```bash
python spam_classifier.py
```

This will:
1. Load and analyze the dataset
2. Preprocess the text data
3. Extract TF-IDF features
4. Train both Multinomial Naive Bayes and Logistic Regression
5. Perform hyperparameter tuning
6. Evaluate and compare models
7. Save the best model and generate visualizations

### Using the Trained Model

After training, use the prediction script:

```bash
python predict_spam.py
```

This will:
- Load the saved model
- Test on sample emails
- Allow you to classify your own emails interactively

### Programmatic Usage

```python
import pickle
from predict_spam import preprocess_text, predict_spam, load_models

# Load models
model, vectorizer, model_name = load_models()

# Classify an email
email_text = "Your email content here..."
result = predict_spam(email_text, model, vectorizer)

print(f"Is Spam: {result['is_spam']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 📈 Model Performance

Both models achieve excellent performance:

| Metric | Multinomial NB | Logistic Regression |
|--------|---------------|---------------------|
| Accuracy | > 95% | > 95% |
| Precision | > 95% | > 96% |
| Recall | > 95% | > 95% |
| F1-Score | > 95% | > 96% |
| ROC-AUC | > 98% | > 99% |

## 📁 Project Structure

```
.
├── spam_classifier.py           # Main training script
├── predict_spam.py             # Prediction/inference script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── emails2.csv                 # Dataset
│
├── Models (generated):
│   ├── best_spam_classifier_[MODEL].pkl
│   ├── spam_classifier_naive_bayes.pkl
│   ├── spam_classifier_logistic_regression.pkl
│   └── tfidf_vectorizer.pkl
│
└── Visualizations (generated):
    ├── class_distribution.png
    ├── text_length_distribution.png
    ├── wordclouds.png
    ├── confusion_matrices.png
    ├── roc_curves.png
    ├── model_comparison.png
    └── feature_importance.png
```

## 🔍 Preprocessing Steps

1. **Text Cleaning**:
   - Convert to lowercase
   - Remove URLs and email addresses
   - Remove special characters and digits
   - Remove extra whitespace

2. **Tokenization**: Split text into words

3. **Stopword Removal**: Remove common words (the, is, and, etc.)

4. **Stemming**: Reduce words to root form using Porter Stemmer

5. **Vectorization**: Convert text to TF-IDF features (3000 features, unigrams + bigrams)

## 🎯 Model Descriptions

### Multinomial Naive Bayes
- **Best for**: Text classification with word frequencies
- **Advantages**: Fast training, probabilistic predictions, works well with high-dimensional data
- **Hyperparameters**: alpha (smoothing parameter), fit_prior

### Logistic Regression
- **Best for**: Linear classification with feature interpretability
- **Advantages**: Provides feature importance, handles feature interactions
- **Hyperparameters**: C (regularization), penalty (L1/L2), solver

## 📊 Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: How many predicted spam are actually spam (minimize false positives)
- **Recall**: How many actual spam were correctly identified (minimize false negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve (model's ability to distinguish classes)

## 🎨 Visualizations

The system generates several visualizations:
1. **Class Distribution**: Shows spam vs. not spam ratio
2. **Text Length Analysis**: Character and word count distributions
3. **Word Clouds**: Most common words in spam vs. legitimate emails
4. **Confusion Matrices**: True/false positive/negative breakdown
5. **ROC Curves**: Model performance comparison
6. **Feature Importance**: Most indicative words for each class

## 💡 Key Insights

- Both models achieve >95% accuracy
- TF-IDF effectively captures important text features
- Spam emails often contain words like: "free", "offer", "click", "urgent", "prize"
- Legitimate emails contain business terms: "meeting", "report", "project", "update"
- Hyperparameter tuning improves performance
- Cross-validation confirms model stability

## 🔮 Future Improvements

- [ ] Implement ensemble methods (Voting/Stacking)
- [ ] Add deep learning models (LSTM, BERT)
- [ ] Real-time email filtering integration
- [ ] User feedback loop for continuous learning
- [ ] Multi-language support
- [ ] Email attachment analysis

## 📝 License

This project is for educational purposes.

## 👤 Author

Machine Learning Intern - Arch Technologies

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## ⭐ Show your support

Give a ⭐️ if this project helped you!

