# 📧 Spam Email Classifier — Naive Bayes & SVM

A machine learning project that classifies SMS/email messages as **Spam** or **Ham (Not Spam)** using NLP techniques — TF-IDF vectorization with Naive Bayes and Linear SVM models.

## 📌 Project Overview

This project uses the UCI SMS Spam Collection dataset to train two classical ML classifiers. It covers end-to-end NLP pipeline: text preprocessing, TF-IDF feature extraction, model training, evaluation, and a custom prediction function.

## 🗂️ Project Structure

```
Spam_Email_Classifier/
│
├── Spam_Email_Classifier.ipynb   # Main notebook
└── README.md
```

## 🧠 Models Used

| Model | Technique |
|---|---|
| Naive Bayes | MultinomialNB with TF-IDF |
| Linear SVM | LinearSVC with TF-IDF |

**Feature Extraction:** TF-IDF with unigrams + bigrams (`ngram_range=(1,2)`), top 10,000 features, English stopwords removed.

## 📊 Results

| Model | Accuracy |
|---|---|
| Naive Bayes | ~98% |
| Linear SVM | ~98.5% |

- **Dataset:** SMS Spam Collection (5,572 messages — 4,825 ham / 747 spam)
- **Source:** Auto-loaded from GitHub URL (no download needed)

## ⚙️ Tech Stack

- Python 3.x
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn

## 🚀 How to Run

**Option 1 — Google Colab (Recommended)**
1. Upload the `.ipynb` file to [Google Colab](https://colab.research.google.com)
2. Click **Runtime → Run All**
3. Dataset loads automatically from URL

**Option 2 — Local**
```bash
pip install scikit-learn pandas matplotlib seaborn
jupyter notebook Spam_Email_Classifier.ipynb
```

> ✅ No manual data download required — dataset loads directly from URL.

## 📈 Notebook Sections

1. Import Libraries
2. Load Dataset (auto from URL)
3. EDA — Class distribution, message length analysis
4. Preprocess & Train-Test Split
5. Model 1 — Naive Bayes Pipeline
6. Model 2 — Linear SVM Pipeline
7. Confusion Matrices (side by side)
8. ROC Curve (Naive Bayes)
9. Custom `predict_spam()` function

## 🔍 Sample Predictions

```
🚨 SPAM  →  Congratulations! You've won a free iPhone. Click now!
✅ HAM   →  Hey, are we still on for lunch tomorrow?
🚨 SPAM  →  URGENT: Your bank account has been compromised. Call now!
✅ HAM   →  Can you send me the notes from today's lecture?
```

## 👤 Author

**Pritam Kumar**  
B.Tech Electrical Engineering, NIT Hamirpur  
GitHub: [github.com/pritam1952](https://github.com/pritam1952)
