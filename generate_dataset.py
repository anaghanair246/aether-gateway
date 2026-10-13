from datasets import load_dataset
import pandas as pd
import csv

print("Fetching 52,000 real-world prompts from Hugging Face...")
dataset = load_dataset("tatsu-lab/alpaca", split="train")

def assign_label(text):
    text_lower = str(text).lower()
    complex_keywords = ['code', 'python', 'c++', 'algorithm', 'calculate', 'analyze', 'equation', 'solve']
    
    if len(str(text).split()) > 35 or any(k in text_lower for k in complex_keywords):
        return 1
    return 0

print("Labeling prompts as 0 (Simple) or 1 (Complex)...")
# Extract texts directly into a list
texts = dataset['instruction']

# Generate labels directly into a list
labels = [assign_label(t) for t in texts]

# Build a pristine dataframe from the two lists
final_df = pd.DataFrame({'text': texts, 'label': labels})

# Export with explicit quoting so commas in the prompts don't break the spreadsheet columns
final_df.to_csv("thesis_dataset.csv", index=False, quoting=csv.QUOTE_MINIMAL)

print(f"Success! Saved {len(final_df)} labeled prompts to 'thesis_dataset.csv'.")
