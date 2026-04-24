import pandas as pd
from transformers import pipeline
import random

# distilgpt2 is fine for a free local model — we just need MORE samples
# and BETTER prompts than the first 50 chars of a Wikipedia article
generator = pipeline('text-generation', model='distilgpt2')

PROMPTS = [
    "Artificial intelligence is transforming the world because",
    "Climate change is one of the most pressing issues because",
    "The history of the internet began when",
    "Electric vehicles are becoming popular because",
    "Machine learning works by",
    "Social media has changed society by",
    "Renewable energy sources such as solar power",
    "The human brain is a complex organ that",
    "Space exploration is important for humanity because",
    "Cryptocurrency is a digital form of money that",
    "Modern smartphones have changed the way people",
    "The global economy depends on",
    "Education systems around the world are changing because",
    "Vaccines work by training the immune system to",
    "The rise of remote work has shown that",
]

print("Generating AI slop samples...")
ai_rows = []

# Keep generating until we match the human dataset size
target = 978
while len(ai_rows) < target:
    prompt = random.choice(PROMPTS)
    try:
        output = generator(
            prompt,
            max_length=200,
            num_return_sequences=1,
            truncation=True,
            do_sample=True,
            temperature=0.9
        )
        ai_text = output[0]['generated_text']
        if len(ai_text) > 100:  # skip very short outputs
            ai_rows.append({
                "text": ai_text,
                "label": 1,
                "source": "distilgpt2"
            })
    except Exception as e:
        print(f"Error: {e}, skipping...")
    
    if len(ai_rows) % 100 == 0:
        print(f"Generated {len(ai_rows)}/{target} samples...")

ai_df = pd.DataFrame(ai_rows)
ai_df.to_csv("ai_slop_data.csv", index=False)
print("Saved ai_slop_data.csv")

# Load your existing human data and merge
human_df = pd.read_csv("human_data.csv")

master_df = pd.concat([human_df, ai_df]).sample(
    frac=1, random_state=42
).reset_index(drop=True)

master_df.to_csv("training_data.csv", index=False)
print(f"\nDone! training_data.csv has {len(master_df)} rows")
print(master_df['label'].value_counts())