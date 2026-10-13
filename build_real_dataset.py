from datasets import load_dataset
import pandas as pd
import csv

print("[1/4] Fetching real complex coding tasks...")
# Dedicated dataset of actual human-written Python tasks and logic problems
complex_ds = load_dataset("flytech/python-codes-25k", split="train")
complex_df = pd.DataFrame(complex_ds)[['instruction']].rename(columns={'instruction': 'text'})
complex_df['label'] = 1

print("[2/4] Fetching standard conversational prompts...")
# Reuse Alpaca for the simple baseline, but filter strictly for short conversational queries
simple_ds = load_dataset("tatsu-lab/alpaca", split="train")
simple_df = pd.DataFrame(simple_ds)[['instruction']].rename(columns={'instruction': 'text'})
simple_df = simple_df[simple_df['text'].str.split().str.len() < 20]
simple_df['label'] = 0

print("[3/4] Forcing a strict 50/50 Class Balance...")
# Take exactly 4,000 of each to prevent the AI from favoring one class
sample_size = 4000
balanced_df = pd.concat([
    simple_df.sample(n=sample_size, random_state=42),
    complex_df.sample(n=sample_size, random_state=42)
])

print("[4/4] Shuffling and saving...")
# Shuffle the rows so the model doesn't process all 0s followed by all 1s
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
balanced_df.to_csv("thesis_dataset_balanced.csv", index=False, quoting=csv.QUOTE_MINIMAL)

print(f"Success! Saved {len(balanced_df)} perfectly balanced rows to 'thesis_dataset_balanced.csv'.")
