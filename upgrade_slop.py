import pandas as pd
import random
import time
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_GROQ_KEY_HERE"
    base_url="https://api.groq.com/openai/v1"
)
MODEL = "llama-3.1-8b-instant"
# client = OpenAI(api_key="YOUR_OPENAI_KEY_HERE")
# MODEL  = "gpt-3.5-turbo"

# ── Diverse prompts — much better than distilgpt2 ─────────────
PROMPTS = [
    "Write a 3-paragraph blog post about climate change and its effects.",
    "Explain how machine learning works to a complete beginner.",
    "Write a news article about the rise of electric vehicles.",
    "Describe the top 5 benefits of daily exercise with explanations.",
    "Write an essay introduction about the impact of social media on society.",
    "Explain cryptocurrency and blockchain in simple terms.",
    "Write a paragraph about the importance of mental health awareness.",
    "Describe how renewable energy is changing the global economy.",
    "Write a short article about recent advances in space exploration.",
    "Explain the basics of personal finance for young adults.",
    "Write a blog post about productivity tips for students.",
    "Describe the history and future of artificial intelligence.",
    "Write a paragraph about the benefits of reading books daily.",
    "Explain how vaccines work and why they are important.",
    "Write an article about the growing importance of cybersecurity.",
    "Describe the impact of smartphones on modern communication.",
    "Write a short essay about the importance of learning to code.",
    "Explain what quantum computing is and why it matters.",
    "Write a blog post about sustainable living and eco-friendly habits.",
    "Describe the role of data science in modern businesses.",
]

def generate_sample(prompt, model=MODEL):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            temperature=0.8
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(5)
        return None

# ── Generate 500 new high-quality slop samples ────────────────
print(f"Generating samples using {MODEL}...")
new_rows = []
target   = 500

while len(new_rows) < target:
    prompt = random.choice(PROMPTS)
    text   = generate_sample(prompt)
    if text and len(text) > 100:
        new_rows.append({
            "text":   text,
            "label":  1,
            "source": MODEL
        })
    if len(new_rows) % 50 == 0 and len(new_rows) > 0 and len(new_rows) != target:
        print(f"Generated {len(new_rows)}/{target}...")
    time.sleep(0.3)  # gentle rate limiting

# ── Merge with existing slop ──────────────────────────────────
existing = pd.read_csv("ai_slop_data.csv")
upgraded = pd.concat(
    [existing, pd.DataFrame(new_rows)],
    ignore_index=True
)
upgraded.to_csv("ai_slop_data.csv", index=False)
print(f"\nSlop dataset upgraded: {len(upgraded)} total samples")
print(upgraded["source"].value_counts().to_string())

# ── Rebuild balanced training_data.csv ───────────────────────
human_df = pd.read_csv("human_data.csv")

# Match human count to new slop count
n = min(len(human_df), len(upgraded))
master_df = pd.concat([
    human_df.sample(n, random_state=42),
    upgraded.sample(n, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

master_df.to_csv("training_data.csv", index=False)
print(f"\nNew training_data.csv: {len(master_df)} rows")
print(master_df["label"].value_counts().to_string())
print("\nStep 1 done. Now run feature_extractor.py")