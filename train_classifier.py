import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import joblib

print("Loading features...")
df = pd.read_csv("training_data_with_features.csv").dropna()
print(f"Rows loaded: {len(df)}")

FEATURE_COLS = [
    "perplexity", "burstiness", "filler_count",
    "avg_word_len", "sentence_count", "vocab_richness"
]

X = df[FEATURE_COLS]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Random Forest...")
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Windows-safe print (no special characters)
print("\nResults:")
print(classification_report(y_test, y_pred,
      target_names=["Human (0)", "AI Slop (1)"]))
print("F1 Score:", round(f1_score(y_test, y_pred), 3))

joblib.dump(clf, "slop_detector.pkl")
print("\nModel saved -> slop_detector.pkl")
print("Phase 2 complete!")