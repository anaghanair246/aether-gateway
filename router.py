import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

class ComplexityRouter:
    def __init__(self, model_path="models/complexity_classifier.onnx"):
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-xsmall")
        
        # Attempt to load the ONNX model
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
            # 1. Tokenize the input prompt
            inputs = self.tokenizer(prompt, return_tensors="np", truncation=True, max_length=128)
            
            # 2. Map to ONNX expected inputs
            onnx_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            }
            
            # 3. Run sub-15ms inference
            outputs = self.session.run(None, onnx_inputs)
            logits = outputs[0]
            
            # 4. Classify (Assuming 0 = cheap, 1 = strong)
            prediction = np.argmax(logits, axis=1)[0]
            selected_tier = "strong" if prediction == 1 else "cheap"
            
        else:
            # Baseline Heuristic
            complex_keywords = ["analyze", "code", "evaluate", "synthesize", "algorithm", "compare"]
            is_complex = len(prompt.split()) > 40 or any(kw in prompt.lower() for kw in complex_keywords)
            selected_tier = "strong" if is_complex else "cheap"

        latency = round((time.perf_counter() - start_time) * 1000, 2)
        
        return {
            "routed_tier": selected_tier,
            "router_latency_ms": latency,
            "routing_mode": self.mode
        }

router = ComplexityRouter()
