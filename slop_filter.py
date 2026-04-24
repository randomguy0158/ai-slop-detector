import pandas as pd
import numpy as np
import math, torch, joblib
import nltk
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Load your saved model ─────────────────────────────────────
print("Loading slop detector model...")
clf       = joblib.load("slop_detector.pkl")
tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
gpt_model = GPT2LMHeadModel.from_pretrained("distilgpt2")
gpt_model.eval()
print("Ready.\n")

# ── Same feature functions from Phase 2 ───────────────────────
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

# ── Core scoring function ─────────────────────────────────────
def get_slop_score(text):
    features = extract_features(text)
    feature_df = pd.DataFrame([features])

    # Probability of being AI (class 1)
    proba = clf.predict_proba(feature_df)[0][1]

    # Convert to 0-100 score
    score = round(proba * 100, 1)

    return score, features

def verdict(score):
    if score >= 75:
        return "FLAGGED — likely AI generated"
    elif score >= 45:
        return "UNCERTAIN — needs manual review"
    else:
        return "CLEAN — likely human written"

# ── Mode 1: Score a single text ───────────────────────────────
def check_text(text):
    print("\n" + "="*55)
    print("INPUT TEXT:")
    print(text[:200] + "..." if len(text) > 200 else text)
    print("-"*55)

    score, features = get_slop_score(text)

    print(f"Slop Score    : {score}/100")
    print(f"Verdict       : {verdict(score)}")
    print("-"*55)
    print(f"Perplexity    : {round(features['perplexity'], 2)}")
    print(f"Burstiness    : {round(features['burstiness'], 3)}")
    print(f"Filler phrases: {features['filler_count']}")
    print(f"Vocab richness: {round(features['vocab_richness'], 3)}")
    print("="*55)
    return score

# ── Mode 2: Filter an entire CSV dataset ─────────────────────
def filter_dataset(input_csv, output_csv, threshold=50):
    print(f"\nLoading {input_csv}...")
    df = pd.read_csv(input_csv).dropna(subset=["text"])
    print(f"Loaded {len(df)} rows")

    scores = []
    for text in df["text"]:
        score, _ = get_slop_score(str(text))
        scores.append(score)

    df["slop_score"] = scores
    df["verdict"]    = df["slop_score"].apply(verdict)

    # Split into clean vs flagged
    clean   = df[df["slop_score"] < threshold]
    flagged = df[df["slop_score"] >= threshold]

    clean.to_csv(output_csv, index=False)

    print(f"\nResults:")
    print(f"Total rows    : {len(df)}")
    print(f"Clean (kept)  : {len(clean)}  ({round(len(clean)/len(df)*100)}%)")
    print(f"Flagged (removed): {len(flagged)} ({round(len(flagged)/len(df)*100)}%)")
    print(f"Saved clean data -> {output_csv}")
    return clean

# ── Run it ────────────────────────────────────────────────────
if __name__ == "__main__":

    # Test 1: Typical AI slop
    check_text("""
    Furthermore, it is important to note that artificial intelligence 
    is transforming industries across the globe. In conclusion, AI 
    represents a paradigm shift that will certainly impact all sectors.
    Moreover, organizations must absolutely embrace these changes.
    """)

    # Test 2: Human-style writing
    check_text("""
    I started learning Python last year after failing my first attempt 
    at C++. It was messy — I kept confusing indentation with brackets. 
    But something clicked when I built a small script to rename my music 
    files. Suddenly it felt real, not just exercises from a textbook.
    """)

    # Test 3: Filter your own training data as a demo
    print("\nRunning filter on training_data.csv as demo...")
    filter_dataset(
        input_csv="training_data.csv",
        output_csv="clean_data.csv",
        threshold=50
    )