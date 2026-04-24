from datasets import load_dataset
import pandas as pd

print("Connecting to Hugging Face...")

# Use the 'wikimedia' version which is the new standard
wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

rows = []
print("Starting to collect articles...")

for i, article in enumerate(wiki):
    if i >= 1000:  # Start with 1000 to test
        break
    
    text = article["text"]
    if len(text) > 500:
        # We save the first 1500 characters as a sample
        rows.append({"text": text[:1500], "label": 0, "source": "wikipedia"})
    
    if i % 100 == 0:
        print(f"Collected {i} articles...")

# Create the CSV
df = pd.DataFrame(rows)
df.to_csv("human_data.csv", index=False)
print("\nSuccess! 'human_data.csv' created.")