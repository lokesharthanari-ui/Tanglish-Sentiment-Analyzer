# Walkthrough - Tanglish Sentiment Analyzer

The **Tanglish Sentiment Analyzer** web application has been successfully built and deployed locally.

## Project Structure Overview

```
d:\TANGlish SENTIMENT ANALYZER\
│
├── app.py                      # Flask backend API with POST /predict & GET /
├── train_model.py              # ML training pipeline (TF-IDF + Logistic Regression)
├── preprocess.py               # Text cleaning & dataset preprocessing module
├── requirements.txt            # Python dependencies (flask, pandas, scikit-learn, joblib)
├── README.md                   # Comprehensive project documentation & viva guide
│
├── dataset/
│   ├── Tamil_first_ready_for_sentiment.csv     # Raw dataset (15,744 rows)
│   └── processed_tanglish_sentiment.csv       # Cleaned prepared dataset (14,187 / 14,242 rows)
│
├── model/
│   ├── sentiment_model.pkl     # Trained Logistic Regression model
│   └── tfidf_vectorizer.pkl    # Trained TF-IDF vectorizer artifact
│
├── templates/
│   └── index.html              # Modern web frontend template
│
└── static/
    ├── style.css               # Glassmorphism dark mode stylesheet
    └── script.js               # Frontend JavaScript with AJAX & animations
```

---

## Preprocessing & ML Model Details

1. **Preprocessing (`preprocess.py`)**:
   - Filtered out unwanted labels (`unknown_state` and `not-Tamil`).
   - Re-mapped `Mixed_feelings` -> `Neutral`.
   - Cleaned text by converting to lowercase, removing URLs, HTML tags, special characters, and duplicate comments.
   - Reduced 15,744 raw records to 14,187 clean prepared comments.

2. **Machine Learning Pipeline (`train_model.py`)**:
   - **Train/Test Split**: 80% Train (11,349 samples), 20% Test (2,838 samples) with stratified sampling.
   - **Feature Extraction**: `TfidfVectorizer(ngram_range=(1, 2), max_features=25000, sublinear_tf=True)`
   - **Classifier**: `LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0)`
   - **Artifacts Saved**: `model/sentiment_model.pkl` and `model/tfidf_vectorizer.pkl`.

---

## Verification & API Results

### Endpoint: `POST /predict`
- **Request Input**: `{"text": "Intha movie semma mass bro"}`
- **Response**:
  ```json
  {
    "sentiment": "Positive",
    "confidence": 0.63,
    "probabilities": {
      "Positive": 0.6262,
      "Negative": 0.2151,
      "Neutral": 0.1587
    },
    "cleaned_text": "intha movie semma mass bro",
    "explanation": "The comment expresses enthusiasm, appreciation, or positive emotion in Tanglish."
  }
  ```

### Example Test Outputs:
- `"Intha movie semma mass bro"` ➔ **Positive** (63% confidence)
- `"Padam romba mokka"` ➔ **Negative** (77% confidence)
- `"Movie okay ah iruku"` ➔ **Neutral** (36% confidence)
- `"Vera level thala acting semma"` ➔ **Positive** (80% confidence)

---

## How to Run the Project Locally

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Train / Retrain Model**:
   ```bash
   python train_model.py
   ```
3. **Launch Flask Server**:
   ```bash
   python app.py
   ```
4. **Access Web App**:
   Open browser at `http://127.0.0.1:5000`
