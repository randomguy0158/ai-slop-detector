# 🛡 Slop Shield — AI Content Detector

A machine learning pipeline that detects AI-generated text and filters it from training datasets to prevent model collapse.

## What it does
- Detects AI-generated "slop" with 99.5% F1 score
- Scores text from 0–100 (clean to flagged)
- Exposes a REST API for integration
- Includes a web UI for easy testing

## How it works
The classifier uses 6 signals:
- **Perplexity** — how predictable is the text?
- **Burstiness** — sentence rhythm variation
- **Filler phrases** — AI giveaway words
- **Vocab richness** — unique word ratio
- **Avg word length** — writing complexity
- **Sentence count** — structural patterns

## Tech stack
- Python, scikit-learn, HuggingFace Transformers
- Flask REST API
- Vanilla HTML/CSS/JS frontend
- Trained on: distilgpt2, Llama 3.1 8B, Llama 3.3 70B

## Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
python api.py

# Open slop_detector_ui.html in browser
```

## API Usage
```bash
# Check a single text
curl -X POST http://127.0.0.1:5000/check \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here"}'

# Response
{
  "slop_score": 99.5,
  "verdict": "FLAGGED - likely AI generated",
  "details": {
    "perplexity": 19.68,
    "burstiness": 0.233,
    "filler_count": 11,
    "vocab_richness": 0.742
  }
}
```

## Results
| Class | Precision | Recall | F1 |
|---|---|---|---|
| Human (0) | 0.99 | 0.99 | 0.99 |
| AI Slop (1) | 0.99 | 0.99 | 0.99 |

**Overall F1: 0.995**

## Built by
Gokul · 2026