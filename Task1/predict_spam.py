"""
Simple script to use the trained spam classifier
Load the saved model and make predictions on new emails
"""

import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download NLTK resources if not already present
try:
    stopwords.words('english')
except:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

def preprocess_text(text):
    """
    Preprocess text for spam detection
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)

def load_models():
    """Load the saved model and vectorizer"""
    try:
        # Try to load Naive Bayes model (usually performs better for spam detection)
        with open('spam_classifier_naive_bayes.pkl', 'rb') as f:
            model = pickle.load(f)
        model_name = "Multinomial Naive Bayes"
    except:
        # Fallback to Logistic Regression
        with open('spam_classifier_logistic_regression.pkl', 'rb') as f:
            model = pickle.load(f)
        model_name = "Logistic Regression"
    
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer, model_name

def predict_spam(email_text, model, vectorizer):
    """
    Predict whether an email is spam or not
    
    Args:
        email_text: The email text to classify
        model: The trained classifier
        vectorizer: The TF-IDF vectorizer
        
    Returns:
        dict: Contains prediction, confidence, and probabilities
    """
    # Preprocess
    processed_email = preprocess_text(email_text)
    
    # Vectorize
    email_tfidf = vectorizer.transform([processed_email])
    
    # Predict
    prediction = model.predict(email_tfidf)[0]
    probability = model.predict_proba(email_tfidf)[0]
    
    return {
        'is_spam': bool(prediction),
        'label': 'SPAM' if prediction == 1 else 'NOT SPAM',
        'confidence': float(probability[prediction]),
        'spam_probability': float(probability[1]),
        'not_spam_probability': float(probability[0])
    }

def main():
    """Main function to demonstrate spam prediction"""
    
    print("="*80)
    print(" "*25 + "SPAM EMAIL CLASSIFIER")
    print("="*80)
    
    # Load models
    print("\nLoading models...")
    model, vectorizer, model_name = load_models()
    print(f"✓ Loaded {model_name} model")
    
    # Test emails
    test_emails = [
        {
            'subject': 'Meeting Tomorrow',
            'body': '''Hi Team,
            
Just a reminder that we have our project status meeting tomorrow at 10 AM.
Please come prepared with your updates.

Thanks,
John'''
        },
        {
            'subject': 'CONGRATULATIONS YOU WON!!!',
            'body': '''Dear Winner,

Congratulations! You have been selected to receive $1,000,000!
Click here to claim your prize NOW! This offer expires in 24 hours!

Act fast! Visit www.scamsite.com and enter your credit card details!'''
        },
        {
            'subject': 'Project Update',
            'body': '''Hello,

The quarterly report is ready for review. Please check the attached document
and provide your feedback by Friday.

Best regards,
Sarah from Accounting'''
        },
        {
            'subject': 'URGENT: Your Account Will Be Suspended',
            'body': '''URGENT NOTICE

Your account will be suspended unless you verify your information immediately!
Click this link and enter your password to verify: http://phishing-site.com

Do not ignore this message!'''
        }
    ]
    
    # Make predictions
    print("\n" + "="*80)
    print("TESTING SPAM DETECTION")
    print("="*80)
    
    for i, email in enumerate(test_emails, 1):
        full_email = f"Subject: {email['subject']}\n\n{email['body']}"
        
        print(f"\n{'='*80}")
        print(f"Email {i}:")
        print(f"{'='*80}")
        print(f"Subject: {email['subject']}")
        print(f"\nBody Preview: {email['body'][:100]}...")
        
        result = predict_spam(full_email, model, vectorizer)
        
        print(f"\n{'─'*80}")
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print(f"Spam Probability: {result['spam_probability']*100:.2f}%")
        print(f"Not Spam Probability: {result['not_spam_probability']*100:.2f}%")
        
        if result['is_spam']:
            print("⚠️  This email appears to be SPAM!")
        else:
            print("✓ This email appears to be legitimate.")
    
    print("\n" + "="*80)
    print("Testing complete!")
    print("="*80)
    
    # Interactive mode
    print("\n\nWould you like to test your own email? (y/n): ", end='')
    try:
        choice = input().lower()
        if choice == 'y':
            print("\nEnter your email text (press Ctrl+D or Ctrl+Z when done):")
            print("-"*80)
            
            try:
                lines = []
                while True:
                    line = input()
                    lines.append(line)
            except EOFError:
                pass
            
            custom_email = '\n'.join(lines)
            
            if custom_email.strip():
                result = predict_spam(custom_email, model, vectorizer)
                
                print("\n" + "="*80)
                print("PREDICTION RESULT:")
                print("="*80)
                print(f"Prediction: {result['label']}")
                print(f"Confidence: {result['confidence']*100:.2f}%")
                print(f"Spam Probability: {result['spam_probability']*100:.2f}%")
                
                if result['is_spam']:
                    print("\n⚠️  This email appears to be SPAM!")
                else:
                    print("\n✓ This email appears to be legitimate.")
    except:
        pass
    
    print("\n\nThank you for using the Spam Email Classifier!")

if __name__ == "__main__":
    main()

