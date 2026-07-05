# 🇵🇰 Urdu Sentiment Classifier

![CI](https://github.com/H-Layba/urdu-sentiment-classifier/actions/workflows/test.yml/badge.svg)
![Model](https://img.shields.io/badge/HuggingFace-H--Layba%2Furdu--sentiment--classifier-yellow)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-Apache%202.0-green)

A fine-tuned **mBERT** model for Urdu sentiment analysis — classifying Urdu text as **positive** or **negative**. Trained on 50,000 Urdu movie reviews with a full MLOps pipeline including CI/CD, automated inference tests, and live deployment on HuggingFace Spaces.

---

## 🚀 Live Demo

👉 **[Try it live on HuggingFace Spaces](https://huggingface.co/spaces/H-Layba/urdu-sentiment-classifier)**

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 81.00% |
| F1 Score (weighted) | 0.8098 |
| Base Model | bert-base-multilingual-cased |
| Training Data | 50,000 Urdu movie reviews |
| Training Epochs | 5 |

### Training Curve

| Epoch | Train Loss | Val Loss | Accuracy | F1 |
|-------|-----------|----------|----------|-----|
| 1 | 0.8954 | 0.9527 | 76.96% | 0.7656 |
| 2 | 0.7918 | 0.8385 | 80.00% | 0.7999 |
| 3 | 0.6211 | 0.8908 | 81.01% | 0.8099 ← best |
| 4 | 0.5833 | 0.9350 | 80.88% | 0.8087 |
| 5 | 0.4515 | 1.0381 | 80.81% | 0.8081 |

---

## 🧪 Example Predictions

| Urdu Text | Translation | Prediction | Confidence |
|-----------|-------------|------------|------------|
| یہ فلم بہت اچھی تھی | This film was very good | ✅ Positive | 99.36% |
| آج کا دن بہت برا تھا | Today was a very bad day | ✅ Negative | 99.18% |
| کھانا لاجواب تھا | The food was excellent | ✅ Positive | 98.9% |
| خدمت بہت خراب تھی | The service was very bad | ✅ Negative | 97.4% |

---

## 🏗️ Project Structure

```
urdu-sentiment-classifier/
├── app.py                    # Gradio demo app (HF Spaces)
├── requirements.txt          # Dependencies
├── README.md
└── .github/
    └── workflows/
        └── test.yml          # CI/CD — auto-runs inference tests on push
└── tests/
    └── test_model.py         # 5 inference quality tests
```

---

## ⚙️ CI/CD Pipeline

Every push to `main` triggers a GitHub Actions workflow that:

1. Installs dependencies
2. Loads the live model from HuggingFace Hub
3. Runs 5 inference tests against known positive/negative sentences
4. Fails the build if any prediction is wrong or confidence drops below 80%

This ensures the deployed model always meets minimum quality standards.

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| Base Model | bert-base-multilingual-cased |
| Fine-tuning | HuggingFace Transformers + Trainer API |
| Training GPU | Kaggle P100 / T4 |
| Training Framework | PyTorch + fp16 mixed precision |
| Experiment Tracking | HuggingFace Trainer logs |
| Demo UI | Gradio |
| Deployment | HuggingFace Spaces |
| CI/CD | GitHub Actions |
| Testing | pytest |

---

## 📦 Use the Model

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="H-Layba/urdu-sentiment-classifier"
)

result = classifier("یہ فلم بہت اچھی تھی")
print(result)
# [{'label': 'positive', 'score': 0.9936}]
```

---

## 📁 Dataset

- **Source:** `mirfan899/imdb_urdu_reviews` on HuggingFace Hub
- **Size:** 50,000 Urdu movie reviews
- **Labels:** Positive (0), Negative (1)
- **Split:** 80% train / 20% test

---

## 🗺️ Roadmap

This is Part 1 of the **Urdu NLP Suite** — a collection of fine-tuned models for Urdu language tasks:

- [x] Sentiment Classification ← you are here
- [ ] Text Summarization (Urdu news articles)
- [ ] Question Answering (Urdu documents)
- [ ] Urdu → English Translation

---

## 👤 Author

**H-Layba**
- HuggingFace: [H-Layba](https://huggingface.co/H-Layba)
- GitHub: [H-Layba](https://github.com/H-Layba)

---

## 📄 License

Apache 2.0
