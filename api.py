from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import math, torch, joblib
import nltk
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from flask_cors import CORS

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

app = Flask(__name__)
CORS(app)

# ── Load model once at startup ────────────────────────────────
print("Loading slop detector...")
clf       = joblib.load("slop_detector.pkl")
tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
gpt_model = GPT2LMHeadModel.from_pretrained("distilgpt2")
gpt_model.eval()
print("API ready.")

# ── Feature functions (same as always) ───────────────────────
def calculate_perplexity(text):
    try:
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=512)
        with torch.no_grad():
            loss = gpt_model(**inputs,
                             labels=inputs["input_ids"]).loss.item()
        return math.exp(loss)
    except:
        return 999.0

def calculate_burstiness(text):
    sentences = nltk.sent_tokenize(text)
    if len(sentences) < 2:
        return 0.0
    lengths   = [len(s.split()) for s in sentences]
    mean, std = np.mean(lengths), np.std(lengths)
    return float(std / mean) if mean > 0 else 0.0

FILLERS = [
    "furthermore", "moreover", "in conclusion",
    "it is important to note", "in summary",
    "to summarize", "as an ai", "as a language model",
    "certainly", "absolutely", "delve", "it is crucial",
    "needless to say", "that being said", "in other words"
]

def count_fillers(text):
    t = text.lower()
    return sum(1 for f in FILLERS if f in t)

def extract_features(text):
    words = text.split()
    return {
        "perplexity":     calculate_perplexity(text),
        "burstiness":     calculate_burstiness(text),
        "filler_count":   count_fillers(text),
        "avg_word_len":   np.mean([len(w) for w in words]) if words else 0,
        "sentence_count": len(nltk.sent_tokenize(text)),
        "vocab_richness": len(set(text.lower().split())) / max(len(words), 1)
    }

def get_slop_score(text):
    features   = extract_features(text)
    feature_df = pd.DataFrame([features])
    proba      = clf.predict_proba(feature_df)[0][1]
    score      = round(proba * 100, 1)
    return score, features

def verdict(score):
    if score >= 75:
        return "FLAGGED - likely AI generated"
    elif score >= 45:
        return "UNCERTAIN - needs manual review"
    else:
        return "CLEAN - likely human written"

# ── Route 1: Health check ─────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status":  "running",
        "service": "AI Slop Detector API",
        "version": "1.0",
        "routes":  ["/check", "/filter"]
    })

# ── Route 2: Check a single text ──────────────────────────────
@app.route("/check", methods=["POST"])
def check():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Send JSON with a 'text' field"}), 400

    text = data["text"]
    if len(text.strip()) < 20:
        return jsonify({"error": "Text too short, minimum 20 characters"}), 400

    score, features = get_slop_score(text)

    return jsonify({
        "slop_score": score,
        "verdict":    verdict(score),
        "details": {
            "perplexity":     round(features["perplexity"],    2),
            "burstiness":     round(features["burstiness"],    3),
            "filler_count":   features["filler_count"],
            "vocab_richness": round(features["vocab_richness"],3)
        }
    })

# ── Route 3: Filter a batch of texts ─────────────────────────
@app.route("/filter", methods=["POST"])
def filter_batch():
    data = request.get_json()

    if not data or "texts" not in data:
        return jsonify({"error": "Send JSON with a 'texts' list"}), 400

    texts     = data["texts"]
    threshold = data.get("threshold", 50)
    results   = []

    for text in texts:
        score, _ = get_slop_score(str(text))
        results.append({
            "text":       text[:100] + "..." if len(text) > 100 else text,
            "slop_score": score,
            "verdict":    verdict(score),
            "kept":       score < threshold
        })

    kept    = [r for r in results if r["kept"]]
    flagged = [r for r in results if not r["kept"]]

    return jsonify({
        "total":    len(results),
        "kept":     len(kept),
        "flagged":  len(flagged),
        "results":  results
    })

# ── Start the server ──────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, port=5000)