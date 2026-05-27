# MLOps Assignment 2 — Goodreads Genre Classification

Fine-tuning **DistilBERT** on the UCSD Goodreads reviews dataset across 8 distinct genres. This project demonstrates an end-to-end MLOps workflow, incorporating experiment tracking via **Weights & Biases (W&B)** and model publishing to the **Hugging Face Hub**.

---

## 🚀 Project Overview

This repository contains a single, production-ready, fully executable notebook (`mlops-assignment-2.ipynb`) designed to train a text classification model on book reviews. It classifies text into one of eight literary genres:
- Children
- Comics & Graphic Novels
- Fantasy & Paranormal
- History & Biography
- Mystery, Thriller & Crime
- Poetry
- Romance
- Young Adult

### Key MLOps Components Included:
- **Streaming & Sampling Data:** Implements efficient streaming of large gzipped JSON files directly from remote servers without downloading the entire dataset locally.
- **Reproducibility:** Seed configurations (`SEED = 42`) across Python, NumPy, and PyTorch ensure stable, reproducible runs.
- **Experiment Tracking:** Real-time logging of training loss, validation loss, accuracy, and F1 scores directly to a **Weights & Biases** dashboard.
- **Artifact Management:** Automatically packages and archives model evaluation reports (`.json` and `.txt`) as a versioned W&B Evaluation Artifact[cite: 1].
- **Model Registry:** Automatically logs in, builds, and pushes the fully fine-tuned transformer weights and tokenizer configurations to a public **Hugging Face Repository**[cite: 1].

---

## 🛠️ Tech Stack & Setup

- **Base Architecture:** `distilbert-base-cased` (via Hugging Face Transformers)[cite: 1]
- **Infrastructure:** Kaggle Notebook (Dual NVIDIA Tesla T4 GPUs)[cite: 1]
- **Tracking System:** Weights & Biases (`wandb`)[cite: 1]
- **Model Storage:** Hugging Face Hub (`huggingface_hub`)[cite: 1]
- **Core Dependencies:** `torch`, `transformers`, `scikit-learn`, `pandas`, `numpy`, `requests`[cite: 1]

### One-Time Environment Configuration
Before executing the notebook, ensure your Kaggle or local environment is configured with the following parameters[cite: 1]:
1. **Accelerator:** GPU T4 x2 (or equivalent CUDA-compatible hardware)[cite: 1].
2. **Internet Access:** Turned **On** (required to stream datasets, authenticate APIs, and push model configurations)[cite: 1].
3. **Secret Environment Variables:**
   - `WANDB_API_KEY`: Found in your [W&B Account Settings](https://wandb.ai/settings)[cite: 1].
   - `HF_TOKEN`: Generated from your [Hugging Face Token Settings](https://huggingface.co/settings/tokens) (Requires **Write** permission)[cite: 1].

---

## 📈 Model Performance & Evaluation

The model was fine-tuned for **3 epochs** using a sequence length of 512 tokens and a learning rate of 3e-5 with FP16 mixed-precision enabled[cite: 1]. 

### Final Metrics Matrix
- **Validation Loss:** `2.2788`[cite: 1]
- **Classification Accuracy:** `60.88%`[cite: 1]
- **Weighted F1-Score:** `0.6096`[cite: 1]

### Per-Class Performance Breakdown

```text
                        precision    recall  f1-score   support

              children       0.70      0.69      0.69       200
        comics_graphic       0.84      0.73      0.78       200
    fantasy_paranormal       0.41      0.48      0.44       200
     history_biography       0.61      0.53      0.56       200
mystery_thriller_crime       0.59      0.69      0.63       200
                poetry       0.75      0.81      0.78       200
               romance       0.61      0.55      0.58       200
           young_adult       0.42      0.40      0.41       200

              accuracy                           0.61      1600
             macro avg       0.61      0.61      0.61      1600
          weighted avg       0.61      0.61      0.61      1600
