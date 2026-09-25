"""
app.py
Tanglish Sentiment Analyzer - Flask REST API Backend

Endpoints:
1. GET / : Renders the web application home page.
2. POST /predict : Accepts JSON {"text": "..."}, preprocesses text, predicts sentiment (Positive/Negative/Neutral),
   and returns predicted sentiment, confidence score, class probabilities, and explanation.
"""

import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from preprocess import clean_text

app = Flask(__name__, template_folder='templates', static_folder='static')

# Global references for model and vectorizer
MODEL = None
VECTORIZER = None

MODEL_PATH = os.path.join("model", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join("model", "tfidf_vectorizer.pkl")


def load_artifacts():
    """Load model and TF-IDF vectorizer artifacts at startup."""
    global MODEL, VECTORIZER
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        try:
            MODEL = joblib.load(MODEL_PATH)
            VECTORIZER = joblib.load(VECTORIZER_PATH)
            print("Successfully loaded model and TF-IDF vectorizer.")
        except Exception as e:
            print(f"Error loading model artifacts: {e}")
    else:
        print("Warning: Model artifacts not found! Please run train_model.py first.")


# Load artifacts when Flask app starts
load_artifacts()


@app.route("/")
def index():
    """Render main web application interface."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    REST API Endpoint to predict sentiment of Tanglish input text.
    Expected JSON Input:
    {
        "text": "intha movie semma mass bro"
    }

    Expected JSON Output:
    {
        "sentiment": "Positive",
        "confidence": 0.94,
        "probabilities": {"Positive": 0.94, "Neutral": 0.04, "Negative": 0.02},
        "cleaned_text": "intha movie semma mass bro",
        "explanation": "..."
    }
    """
    if not MODEL or not VECTORIZER:
        return jsonify({
            "error": "Model not loaded. Please ensure train_model.py has been executed."
        }), 500

    try:
        data = request.get_json(force=True, silent=True)
        if not data or "text" not in data:
            return jsonify({"error": "Invalid request. Please provide 'text' key in JSON body."}), 400

        raw_text = data.get("text", "")
        if not raw_text or not str(raw_text).strip():
            return jsonify({"error": "Input text cannot be empty."}), 400

        # Preprocess input text using standardized preprocess.py clean_text
        cleaned = clean_text(raw_text)
        if not cleaned:
            return jsonify({
                "sentiment": "Neutral",
                "confidence": 0.50,
                "probabilities": {"Positive": 0.33, "Neutral": 0.50, "Negative": 0.17},
                "cleaned_text": "",
                "explanation": "No significant words detected after cleaning."
            })

        # Feature vectorization
        text_vectorized = VECTORIZER.transform([cleaned])

        # Predict sentiment and probability distribution
        predicted_category = MODEL.predict(text_vectorized)[0]
        probabilities = MODEL.predict_proba(text_vectorized)[0]
        classes = list(MODEL.classes_)

        prob_map = {cls: round(float(prob), 4) for cls, prob in zip(classes, probabilities)}
        confidence = prob_map.get(predicted_category, round(float(max(probabilities)), 4))

        # Generate human-readable explanation
        explanation_map = {
            "Positive": "The comment expresses enthusiasm, appreciation, or positive emotion in Tanglish.",
            "Negative": "The comment expresses dissatisfaction, criticism, or negative sentiment in Tanglish.",
            "Neutral": "The comment presents a balanced, indifferent, or mixed perspective in Tanglish."
        }
        explanation = explanation_map.get(
            predicted_category, "Sentiment evaluated based on Tanglish NLP features."
        )

        response_payload = {
            "sentiment": str(predicted_category),
            "confidence": round(float(confidence), 2),
            "probabilities": prob_map,
            "cleaned_text": cleaned,
            "explanation": explanation
        }

        return jsonify(response_payload), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    """Health check and model status endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "vectorizer_loaded": VECTORIZER is not None
    })


if __name__ == "__main__":
    print("Starting Tanglish Sentiment Analyzer Flask Server...")
    app.run(host="0.0.0.0", port=5000, debug=True)
