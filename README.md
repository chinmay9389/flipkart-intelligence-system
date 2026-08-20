# Flipkart Order Intelligence & Support Assistant System

An end-to-end connected intelligent system combining return-risk machine learning, product image transfer-learning classification, and a grounded LangGraph support assistant for Flipkart operations.

---

## System Architecture & Overview
1. **Part 1 (Return-Risk Scoring Pipeline):** Random Forest model trained on Flipkart dataset.
2. **Part 2 (Product Image Categoriser):** ResNet-18 vision model trained on Fashion-MNIST.
3. **Part 3 (Support Agent & Grounded RAG):** LangGraph agent with tool execution and policy RAG.

---

## Reproduction & Execution Guide

### Environment Setup
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn torch torchvision sentence-transformers faiss-cpu langgraph
```

### Execution Steps
```bash
python3 generate_orders.py
python3 train_part1.py
python3 train_part2.py
python3 eval_and_transcripts.py
```
