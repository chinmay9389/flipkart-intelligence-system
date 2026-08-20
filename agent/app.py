import json
import re
import os
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from agent.tools import check_return_risk, classify_product_image

# 1. RAG Vector Index Setup
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
kb_path = os.path.join(base_dir, "policy_kb", "documents.json")

with open(kb_path) as f:
    DOCUMENTS = json.load(f)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_texts = [d["text"] for d in DOCUMENTS]
embeddings = embedder.encode(doc_texts, convert_to_numpy=True)

dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(embeddings)
faiss_index.add(embeddings)

def retrieve_policy(query: str, top_k=3, threshold=0.35):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, indices = faiss_index.search(q_emb, top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "doc_id": DOCUMENTS[idx]["doc_id"],
            "text": DOCUMENTS[idx]["text"],
            "score": float(score)
        })
    return results

# 2. Guardrails
INJECTION_PATTERNS = ["ignore previous instructions", "ignore all rules", "pretend you are"]

def check_input_guardrail(query: str) -> bool:
    for pattern in INJECTION_PATTERNS:
        if pattern.lower() in query.lower():
            return False
    return True

# 3. Deterministic Mock LLM Graph Response Generator
def run_flipkart_agent(user_query: str, order_context: dict = None, image_path: str = None) -> Dict[str, Any]:
    # Guardrail check
    if not check_input_guardrail(user_query):
        return {
            "answer": "Security Alert: Prompt injection pattern detected. Query blocked by input guardrail.",
            "source": "guardrail_block",
            "confidence": 0.0
        }
    
    # Intent Detection Simulation (Few-shot driven logic)
    query_lower = user_query.lower()
    
    if "return risk" in query_lower or "likely to be returned" in query_lower or order_context is not None:
        if not order_context:
            order_context = {
                "product_category": "Apparel", "price_inr": 1500.0, "discount_pct": 25.0,
                "payment_method": "COD", "customer_tenure_days": 120, "num_previous_orders": 3,
                "num_previous_returns": 2, "delivery_distance_km": 450.0, "delivery_days": 5,
                "is_weekend_order": 1, "rating_given": 4.0
            }
        risk_res = check_return_risk(order_context)
        return {
            "answer": f"Order return risk assessed. Predicted Return Risk Probability is {risk_res['return_probability']*100:.1f}% ({risk_res['risk_bucket']} Risk Bucket).",
            "source": "return_risk_tool",
            "confidence": 0.95,
            "tool_details": risk_res
        }
        
    elif "image" in query_lower or "category" in query_lower or image_path is not None:
        path = image_path if image_path else "data/sample_images/04_shirt.png"
        img_res = classify_product_image(path)
        return {
            "answer": f"Image classified successfully. Visual product category identified as '{img_res['predicted_category']}' with {img_res['confidence']*100:.1f}% confidence.",
            "source": "image_classifier_tool",
            "confidence": img_res["confidence"],
            "tool_details": img_res
        }
        
    else:
        # Policy RAG Routing
        retrieved = retrieve_policy(user_query, top_k=3)
        top_score = retrieved[0]["score"] if retrieved else 0.0
        
        # Groundedness output guardrail check
        if top_score < 0.35:
            return {
                "answer": f"Refusal: I cannot answer this policy question as no grounded policy clear enough was found (Highest Similarity Score: {top_score:.4f} below threshold 0.35).",
                "source": "policy_kb_refusal",
                "confidence": round(top_score, 4)
            }
            
        return {
            "answer": retrieved[0]["text"],
            "source": "policy_kb",
            "confidence": round(top_score, 4)
        }
