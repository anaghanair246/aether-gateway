import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

print("Loading ONNX model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
session = ort.InferenceSession("models/complexity_classifier.onnx")

test_prompts = [
    "Hello!",
    "What is the capital of France?",
    "Write a Python script using dynamic programming to solve the 0/1 knapsack problem.",
    "Develop a comprehensive Python application using FastAPI, SQLAlchemy, and Celery to build a microservice."
]

print("\n--- RAW MODEL LOGITS ---")
for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="np", truncation=True, padding="max_length", max_length=128)
    
    onnx_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }
    
    # Get the raw mathematical outputs
    logits = session.run(None, onnx_inputs)[0][0]
    prediction = np.argmax(logits)
    
    # Print the raw numbers
    print(f"\nPrompt: {prompt[:40]}...")
    print(f"Logits [Class 0, Class 1]: {logits}")
    print(f"Final Prediction: {prediction}")
