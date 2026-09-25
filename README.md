# Tanglish Sentiment Analyzer Using NLP and Machine Learning

A complete Natural Language Processing (NLP) and Machine Learning based web application designed to analyze sentiment expressed in **Tanglish** (Tamil language transliterated in English script).

---

## 📌 Abstract

**Tanglish** (a blend of Tamil and English) is extensively used across social media, YouTube comments, online forums, and digital messaging. Standard English or Tamil sentiment analyzers fail to accurately classify Tanglish due to spelling variations, code-mixing, informal slang, and phonetically transliterated words.

This project solves this problem by building a dedicated sentiment analysis system. Using a labelled Tamil-English dataset of **15,744 initial comments**, the pipeline filters out invalid classes (`unknown_state`, `not-Tamil`) and maps `Mixed_feelings` to `Neutral`. After text normalization, cleaning, and duplicate removal, the final prepared dataset contains **14,242 comments**.

Text feature extraction is performed using **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization with unigrams and bigrams. A **Logistic Regression** machine learning model is trained on 80% of the dataset and evaluated on 20% stratified test data. The system is deployed as a responsive web application using a **Python Flask backend** and an attractive **HTML/CSS/JS frontend**.

---

## 📁 Project Directory Structure

```
tanglish-sentiment-analyzer/
│
├── app.py                      # Flask Web Application & REST API
├── train_model.py              # ML Training & Evaluation Pipeline
├── preprocess.py               # Text Preprocessing & Dataset Cleaning
├── requirements.txt            # Python Dependencies
├── README.md                   # Project Documentation
│
├── dataset/
│   ├── Tamil_first_ready_for_sentiment.csv     # Raw Dataset
│   └── processed_tanglish_sentiment.csv       # Cleaned Dataset (14,242 records)
│
├── model/
│   ├── sentiment_model.pkl     # Trained Logistic Regression Model
│   └── tfidf_vectorizer.pkl    # Trained TF-IDF Vectorizer
│
├── templates/
│   └── index.html              # HTML5 Web UI Template
│
└── static/
    ├── style.css               # Vanilla CSS Styling & Glassmorphism Theme
    └── script.js               # Frontend JavaScript & AJAX Fetch API
```

---

## 📊 Dataset & Preprocessing Details

- **Original Dataset Count**: 15,744 records
- **Original Classes**: `Positive`, `Negative`, `Mixed_feelings`, `unknown_state`, `not-Tamil`
- **Class Filtering**:
  - Removed `unknown_state` (850 records)
  - Removed `not-Tamil` (497 records)
  - Re-mapped `Mixed_feelings` -> `Neutral` (1,777 records)
- **Prepared Dataset Count**: **14,242 records**
  - **Positive**: 10,385 comments
  - **Negative**: 2,025 comments
  - **Neutral**: 1,777 comments

### Preprocessing Operations (`preprocess.py`):
1. **Lowercasing**: Standardizes case sensitivity.
2. **URL Removal**: Strips `http://`, `https://`, and `www` links.
3. **HTML Tag Stripping**: Removes HTML elements.
4. **Special Character Removal**: Cleans emojis and non-alphanumeric noise while preserving Tanglish words.
5. **Whitespace Normalization**: Eliminates double spaces and trailing spaces.
6. **Duplicate Removal**: Removes redundant cleaned comments to prevent data leakage.

---

## 🤖 Machine Learning Pipeline (`train_model.py`)

- **Train-Test Split**: 80% Training Data, 20% Test Data (Stratified Sampling).
- **Feature Extraction**: `TfidfVectorizer(ngram_range=(1, 2), max_features=25000, sublinear_tf=True)`
- **Classifier Model**: `LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0)`
- **Model Storage**: Saved using `joblib` inside `model/` directory.

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess & Train the Model
```bash
python train_model.py
```
*Outputs accuracy metrics, classification report, confusion matrix, and generates `model/sentiment_model.pkl` and `model/tfidf_vectorizer.pkl`.*

### 3. Run the Flask Web Application
```bash
python app.py
```
Open your browser and navigate to: **`http://localhost:5000`**

---

## 🌐 REST API Endpoint Specification

### `POST /predict`

#### Request JSON:
```json
{
  "text": "intha movie semma mass bro"
}
```

#### Response JSON:
```json
{
  "sentiment": "Positive",
  "confidence": 0.94,
  "probabilities": {
    "Positive": 0.94,
    "Neutral": 0.04,
    "Negative": 0.02
  },
  "cleaned_text": "intha movie semma mass bro",
  "explanation": "The comment expresses enthusiasm, appreciation, or positive emotion in Tanglish."
}
```

---

## 💡 Viva & Project Demonstration Highlights

1. **Why Logistic Regression & TF-IDF?**
   - Logistic Regression provides fast, interpretable probabilistic outputs (`predict_proba`) well-suited for sparse text matrices produced by TF-IDF.
   - Using **Unigrams and Bigrams** (`ngram_range=(1, 2)`) captures word combinations like `"semma mass"`, `"romba mokka"`, and `"okay ah"` effectively.
2. **Handling Class Imbalance**:
   - `class_weight='balanced'` adjusts weights inversely proportional to class frequencies, improving recall for minority classes (`Negative` & `Neutral`).
3. **Real-World Applications**:
   - YouTube comment sentiment analysis for regional Tamil movies/trailers.
   - E-commerce customer feedback analysis for Tamil-speaking regions.
   - Social media brand monitoring.
