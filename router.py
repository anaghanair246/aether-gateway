import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

def softmax(x):
    """Converts raw logits into percentages/probabilities (0.0 to 1.0)"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

class ComplexityRouter:
    def __init__(self, model_path="models/complexity_classifier.onnx"):
        # Swapped to DistilBERT to match our stable ONNX model
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
        try:
            self.session = ort.InferenceSession(model_path)
            self.mode = "onnx_ml"
            print(f"[Router] Initialized ONNX ML Engine from {model_path}")
        except FileNotFoundError:
            self.mode = "heuristic_fallback"
            print(f"[Router] ONNX model not found at {model_path}. Using heuristic fallback.")

    def predict_tier(self, prompt: str) -> dict:
        start_time = time.perf_counter()
        
        if self.mode == "onnx_ml":
            inputs = self.tokenizer(
                prompt, 
                return_tensors="np", 
                truncation=True, 
                padding="max_length", 
                max_length=128
            )
            
            onnx_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            }
            
            outputs = self.session.run(None, onnx_inputs)
            raw_logits = outputs[0][0]
            
            # Thesis Metrics: Calculate confidence probabilities
            probs = softmax(raw_logits)
            confidence = float(np.max(probs))
            predicted_class = int(np.argmax(probs))
            
            selected_tier = "strong" if predicted_class == 1 else "cheap"
            logits_list = [round(float(x), 4) for x in raw_logits]
            
        else:
            complex_keywords = ["analyze", "code", "evaluate", "synthesize", "algorithm", "compare"]
            is_complex = len(prompt.split()) > 40 or any(kw in prompt.lower() for kw in complex_keywords)
            selected_tier = "strong" if is_complex else "cheap"
            confidence = 1.0
            logits_list = []

        latency = round((time.perf_counter() - start_time) * 1000, 2)
        
        return {
            "routed_tier": selected_tier,
            "confidence": round(confidence, 4),
            "logits": logits_list,
            "router_latency_ms": latency,
            "routing_mode": self.mode
        }

router = ComplexityRouter()
