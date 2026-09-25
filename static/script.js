/**
 * script.js
 * Tanglish Sentiment Analyzer - Client Side Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const tanglishInput = document.getElementById('tanglishInput');
    const charCount = document.getElementById('charCount');
    const clearBtn = document.getElementById('clearBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resetBtn = document.getElementById('resetBtn');
    const spinner = document.getElementById('spinner');
    
    // Result Card Elements
    const resultPlaceholder = document.getElementById('resultPlaceholder');
    const resultContent = document.getElementById('resultContent');
    const confidenceVal = document.getElementById('confidenceVal');
    const sentimentBanner = document.getElementById('sentimentBanner');
    const sentimentIcon = document.getElementById('sentimentIcon');
    const sentimentTitle = document.getElementById('sentimentTitle');
    const sentimentDesc = document.getElementById('sentimentDesc');
    const origTextVal = document.getElementById('origTextVal');
    const cleanedTextVal = document.getElementById('cleanedTextVal');
    
    // Probability Elements
    const probPosVal = document.getElementById('probPosVal');
    const probNeuVal = document.getElementById('probNeuVal');
    const probNegVal = document.getElementById('probNegVal');
    const barPos = document.getElementById('barPos');
    const barNeu = document.getElementById('barNeu');
    const barNeg = document.getElementById('barNeg');

    // Preset Chip Buttons
    const exampleChips = document.querySelectorAll('.chip');

    // 1. Textarea Character Count & Clear Button Visibility
    tanglishInput.addEventListener('input', () => {
        const val = tanglishInput.value;
        charCount.textContent = `${val.length} / 500 chars`;
        clearBtn.style.display = val.length > 0 ? 'block' : 'none';
    });

    // Clear Button Click
    clearBtn.addEventListener('click', () => {
        tanglishInput.value = '';
        charCount.textContent = '0 / 500 chars';
        clearBtn.style.display = 'none';
        tanglishInput.focus();
    });

    // 2. Preset Example Chips Click Handler
    exampleChips.forEach(chip => {
        chip.addEventListener('click', () => {
            const presetText = chip.getAttribute('data-text');
            tanglishInput.value = presetText;
            charCount.textContent = `${presetText.length} / 500 chars`;
            clearBtn.style.display = 'block';
            
            // Automatically analyze selected example
            analyzeSentiment();
        });
    });

    // 3. Reset Button Click Handler
    resetBtn.addEventListener('click', () => {
        tanglishInput.value = '';
        charCount.textContent = '0 / 500 chars';
        clearBtn.style.display = 'none';
        
        // Reset Result View
        resultContent.classList.add('hidden');
        resultPlaceholder.classList.remove('hidden');
    });

    // 4. Analyze Button Click Handler
    analyzeBtn.addEventListener('click', analyzeSentiment);

    // Enter key (Shift+Enter for newline, Enter to submit)
    tanglishInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            analyzeSentiment();
        }
    });

    /**
     * Function to call POST /predict REST API
     */
    async function analyzeSentiment() {
        const text = tanglishInput.value.trim();
        if (!text) {
            alert('Please enter or select a Tanglish text comment to analyze.');
            tanglishInput.focus();
            return;
        }

        // UI Loading State
        analyzeBtn.disabled = true;
        spinner.style.display = 'inline-block';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            });

            const data = await response.json();

            if (!response.ok || data.error) {
                alert(data.error || 'An error occurred while analyzing sentiment.');
                return;
            }

            // Display Predictions
            displayResult(text, data);

        } catch (err) {
            console.error('Fetch error:', err);
            alert('Unable to connect to the backend server. Please check if Flask is running.');
        } finally {
            // Restore UI state
            analyzeBtn.disabled = false;
            spinner.style.display = 'none';
        }
    }

    /**
     * Update Result Card UI
     */
    function displayResult(originalText, data) {
        const sentiment = data.sentiment; // "Positive", "Negative", "Neutral"
        const confidencePct = Math.round(data.confidence * 100);
        const probs = data.probabilities || {};

        // Update Text Breakdowns
        origTextVal.textContent = `"${originalText}"`;
        cleanedTextVal.textContent = data.cleaned_text ? `"${data.cleaned_text}"` : "(No significant text)";

        // Update Confidence
        confidenceVal.textContent = `${confidencePct}%`;

        // Reset Banner Classes
        sentimentBanner.className = 'sentiment-banner';
        
        // Icon and Theme Configurations
        if (sentiment === 'Positive') {
            sentimentBanner.classList.add('positive');
            sentimentIcon.className = 'fa-solid fa-face-smile';
            sentimentTitle.textContent = 'Positive Sentiment';
        } else if (sentiment === 'Negative') {
            sentimentBanner.classList.add('negative');
            sentimentIcon.className = 'fa-solid fa-face-frown';
            sentimentTitle.textContent = 'Negative Sentiment';
        } else {
            sentimentBanner.classList.add('neutral');
            sentimentIcon.className = 'fa-solid fa-face-meh';
            sentimentTitle.textContent = 'Neutral Sentiment';
        }

        sentimentDesc.textContent = data.explanation || 'Sentiment analyzed using trained NLP Logistic Regression model.';

        // Update Probability Bars
        const posProb = Math.round((probs.Positive || 0) * 100);
        const neuProb = Math.round((probs.Neutral || 0) * 100);
        const negProb = Math.round((probs.Negative || 0) * 100);

        probPosVal.textContent = `${posProb}%`;
        probNeuVal.textContent = `${neuProb}%`;
        probNegVal.textContent = `${negProb}%`;

        // Animate width
        setTimeout(() => {
            barPos.style.width = `${posProb}%`;
            barNeu.style.width = `${neuProb}%`;
            barNeg.style.width = `${negProb}%`;
        }, 100);

        // Switch View
        resultPlaceholder.classList.add('hidden');
        resultContent.classList.remove('hidden');
    }
});
