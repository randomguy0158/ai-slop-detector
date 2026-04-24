import pandas as pd
import numpy as np
import math, torch
import nltk
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import joblib

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Load distilgpt2 (already downloaded on your machine) ────
print("Loading distilgpt2...")
tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
model     = GPT2LMHeadModel.from_pretrained("distilgpt2")
model.eval()

# ── Signal 1: Perplexity ─────────────────────────────────────
def calculate_perplexity(text):
    try:
        inputs = tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=512
        )
        with torch.no_grad():
            loss = model(**inputs, labels=inputs["input_ids"]).loss.item()
        return math.exp(loss)
    except:
        return 999.0

# ── Signal 2: Burstiness ─────────────────────────────────────
def calculate_burstiness(text):
    sentences = nltk.sent_tokenize(text)
    if len(sentences) < 2:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    mean, std = np.mean(lengths), np.std(lengths)
    return float(std / mean) if mean > 0 else 0.0

# ── Signal 3: Filler phrases ─────────────────────────────────
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

# ── All features combined ─────────────────────────────────────
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

# ── Load your training_data.csv ───────────────────────────────
print("Loading training_data.csv...")
df = pd.read_csv("training_data.csv").dropna(subset=["text"])
print(f"Loaded {len(df)} rows — {df['label'].value_counts().to_dict()}")

# ── Extract features for every row ───────────────────────────
print("\nExtracting features (will take ~20-40 mins on CPU)...")
features = []
for text in tqdm(df["text"], desc="Processing"):
    features.append(extract_features(str(text)))

feature_df = pd.DataFrame(features)
df = pd.concat([df.reset_index(drop=True), feature_df], axis=1)
df.to_csv("training_data_with_features.csv", index=False)
print("Saved training_data_with_features.csv")

# ── Train the classifier ──────────────────────────────────────
print("\nTraining Random Forest classifier...")
FEATURE_COLS = [
    "perplexity", "burstiness", "filler_count",
    "avg_word_len", "sentence_count", "vocab_richness"
]

X = df[FEATURE_COLS]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# ── Results ───────────────────────────────────────────────────
y_pred = clf.predict(X_test)
print("\n── Results ──────────────────────────────────")
print(classification_report(y_test, y_pred,
      target_names=["Human (0)", "AI Slop (1)"]))
print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")

# ── Save the model ────────────────────────────────────────────
joblib.dump(clf, "slop_detector.pkl")
print("\nModel saved → slop_detector.pkl")
print("Phase 2 complete!")