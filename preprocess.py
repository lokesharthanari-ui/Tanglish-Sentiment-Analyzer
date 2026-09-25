"""
preprocess.py
Tanglish Sentiment Analyzer - Data Preprocessing Module

This module handles:
1. Text cleaning operations (lowercasing, removing URLs, HTML tags, special characters, extra spaces).
2. Dataset preprocessing (filtering categories, handling missing values, duplicate removal, label mapping).
"""

import os
import re
import pandas as pd


def clean_text(text: str) -> str:
    """
    Preprocess individual Tanglish text string.
    - Convert to lowercase
    - Remove URLs
    - Remove HTML tags
    - Remove unwanted special characters/emojis while preserving letters & numbers
    - Normalize whitespace
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. Remove HTML tags
    text = re.sub(r'<.*?>+', '', text)
    
    # 4. Remove special characters/emojis (keep word chars & whitespace)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 5. Replace underscores with space (since \w includes _)
    text = text.replace('_', ' ')
    
    # 6. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def load_and_preprocess_dataset(input_path: str = None, output_path: str = None) -> pd.DataFrame:
    """
    Load raw Tanglish dataset, perform filtering & cleaning, and save processed CSV.
    """
    if input_path is None:
        input_path = os.path.join("dataset", "Tamil_first_ready_for_sentiment.csv")
    if output_path is None:
        output_path = os.path.join("dataset", "processed_tanglish_sentiment.csv")
    
    print(f"Loading raw dataset from: {input_path}")
    # Raw file is tab-separated without headers
    df = pd.read_csv(input_path, sep='\t', header=None, names=['category', 'text'])
    
    initial_count = len(df)
    print(f"Original record count: {initial_count}")
    
    # Clean category string formatting
    df['category'] = df['category'].astype(str).str.strip()
    
    # Exclude unknown_state and not-Tamil categories
    excluded_labels = ['unknown_state', 'not-Tamil']
    df = df[~df['category'].isin(excluded_labels)].copy()
    print(f"Record count after removing {excluded_labels}: {len(df)}")
    
    # Map Mixed_feelings to Neutral
    df['category'] = df['category'].replace({'Mixed_feelings': 'Neutral'})
    
    # Handle missing values
    df = df.dropna(subset=['text']).copy()
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'] != ''].copy()
    
    # Apply clean_text
    print("Applying text preprocessing...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Remove rows where cleaned text is empty
    df = df[df['cleaned_text'] != ''].copy()
    
    # Remove duplicate records based on cleaned text
    df = df.drop_duplicates(subset=['cleaned_text']).copy()
    final_count = len(df)
    
    print(f"Final prepared record count: {final_count}")
    print("\nClass distribution:")
    print(df['category'].value_counts())
    
    # Ensure directory exists and save processed dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nProcessed dataset saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    load_and_preprocess_dataset()
