"""
train_model.py
Tanglish Sentiment Analyzer - Machine Learning Model Training Script

This script:
1. Loads preprocessed dataset (or runs preprocess.py if missing).
2. Performs 80-20 stratified train-test split.
3. Applies TF-IDF vectorization (unigrams + bigrams).
4. Trains Logistic Regression classifier with balanced class weights.
5. Evaluates model performance (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
6. Saves model artifacts (sentiment_model.pkl, tfidf_vectorizer.pkl).
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from preprocess import load_and_preprocess_dataset


def train_tanglish_sentiment_model():
    dataset_path = os.path.join("dataset", "processed_tanglish_sentiment.csv")
    
    # Load dataset if it exists, otherwise generate it
    if not os.path.exists(dataset_path):
        print("Processed dataset not found. Running preprocessing first...")
        df = load_and_preprocess_dataset()
    else:
        print(f"Loading preprocessed dataset from: {dataset_path}")
        df = pd.read_csv(dataset_path)
    
    # Drop any nulls if present
    df = df.dropna(subset=['cleaned_text', 'category']).copy()
    
    X = df['cleaned_text']
    y = df['category']
    
    print(f"\nDataset shape: {df.shape}")
    print("Class counts:")
    print(y.value_counts())
    
    # Step 1: Stratified Train-Test Split (80% train, 20% test)
    print("\n--- Step 1: Splitting Data (80% Train, 20% Test) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    
    # Step 2: TF-IDF Vectorization
    print("\n--- Step 2: TF-IDF Vectorization (Unigrams & Bigrams) ---")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=25000,
        sublinear_tf=True,
        min_df=1
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"TF-IDF feature shape: {X_train_tfidf.shape}")
    
    # Step 3: Logistic Regression Classifier
    print("\n--- Step 3: Training Logistic Regression Model ---")
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        C=1.0,
        random_state=42
    )
    
    model.fit(X_train_tfidf, y_train)
    print("Model training completed successfully.")
    
    # Step 4: Model Evaluation
    print("\n--- Step 4: Model Evaluation ---")
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc * 100:.2f}%\n")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    
    labels = model.classes_
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Confusion Matrix:")
    cm_df = pd.DataFrame(cm, index=[f"Actual_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels])
    print(cm_df)
    
    # Step 5: Save Model Artifacts
    print("\n--- Step 5: Saving Model Artifacts ---")
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "sentiment_model.pkl")
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"Saved sentiment model to: {model_path}")
    print(f"Saved TF-IDF vectorizer to: {vectorizer_path}")
    
    print("\nModel training pipeline finished successfully!")
    return model, vectorizer, acc


if __name__ == "__main__":
    train_tanglish_sentiment_model()
