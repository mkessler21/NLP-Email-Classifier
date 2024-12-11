import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, auc
from sklearn.model_selection import GridSearchCV
from collections import Counter
import re

# Visualization settings
sns.set(style="whitegrid", color_codes=True, font_scale=1.5)

def load_and_preprocess_data():
    """Load and preprocess the email data"""
    import zipfile
    with zipfile.ZipFile('spam_ham_data.zip') as item:
        item.extractall()

    original_training_data = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # Convert emails to lowercase
    original_training_data['email'] = original_training_data['email'].str.lower()
    test['email'] = test['email'].str.lower()

    # Handle missing values
    original_training_data = original_training_data.fillna('')
    test = test.fillna('')

    return original_training_data, test

def words_in_texts(words, texts):
    """Create binary features for word presence in texts"""
    return np.array([[int(word in text) for word in words] for text in texts])

def keyword_indicator(words, text):
    """Check presence of keywords in text"""
    return [1 if word in text else 0 for word in words]

def count_exclamations(text):
    """Count exclamation marks in text"""
    return text.count('!')

def count_uppercase(text):
    """Count uppercase characters in text"""
    text = str(text)
    return sum(1 for c in text if c.isupper())

def extract_features(data, words):
    """Extract combined features from emails"""
    keyword_features = np.array([keyword_indicator(words, email) for email in data['email']])
    exclamation_features = np.array(data['email'].apply(count_exclamations)).reshape(-1, 1)
    uppercase_features = np.array(data['subject'].apply(count_uppercase)).reshape(-1, 1)
    return np.hstack((keyword_features, exclamation_features, uppercase_features))

def plot_word_correlations(train_data, selected_words):
    """Plot correlation matrix of word features"""
    word_indicator_matrix = words_in_texts(selected_words, train_data['email'])
    word_presence_df = pd.DataFrame(word_indicator_matrix, columns=selected_words)
    word_presence_df['type'] = train_data['spam'].replace({0: 'Ham', 1: 'Spam'})
    word_presence_df['type'] = word_presence_df['type'].map({'Ham': 0, 'Spam': 1})
    
    correlation_matrix = word_presence_df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Word Indicators and Email Type')
    plt.show()

def plot_roc_curve(model, features, labels):
    """Plot ROC curve for model performance"""
    y_pred_prob = model.predict_proba(features)[:, 1]
    fpr, tpr, thresholds = roc_curve(labels, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='red', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

def main():
    # Load and prepare data
    original_training_data, test = load_and_preprocess_data()
    
    # Split into train/validation sets
    train, val = train_test_split(original_training_data, test_size=0.1, random_state=42)
    train = train.reset_index(drop=True)

    # Define feature words
    words = ['html', 'please', 'body', 'free', 'click', 'font', 'br', 'head']

    # Extract features
    train_features = extract_features(train, words)
    test_features = extract_features(test, words)

    # Train model with grid search
    logistic_model = LogisticRegression(max_iter=1000)
    parameters = {'C': [0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(logistic_model, parameters)
    grid_search.fit(train_features, train['spam'])

    # Make predictions
    train_predictions = grid_search.predict(train_features)
    test_predictions = grid_search.predict(test_features)

    # Calculate accuracy
    training_accuracy = accuracy_score(train['spam'], train_predictions)
    print(f"Training Accuracy: {training_accuracy:.4f}")

    # Plot word correlations
    plot_word_correlations(train, words)

    # Plot ROC curve
    plot_roc_curve(grid_search.best_estimator_, train_features, train['spam'])

    # Save test predictions
    submission_df = pd.DataFrame({
        "Id": test['id'],
        "Class": test_predictions
    }, columns=['Id', 'Class'])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"submission_{timestamp}.csv"
    submission_df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")

if __name__ == "__main__":
    main()