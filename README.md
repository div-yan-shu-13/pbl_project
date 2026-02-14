# Explainable Multi-Class Mental Health Classification and Linguistic Differentiation Across Reddit Communities

This project investigates multi-class classification of Reddit mental health communities using classical machine learning and transformer-based models, followed by explainability analysis to understand model behavior.

The objective is to analyze linguistic differences between mental health subreddits and examine why contextual models confuse certain conditions such as Depression and SuicideWatch.

---

## Project Structure

The repository contains three main notebooks:

### 1. `baseline.ipynb`

* Loads Reddit dataset (2018 + 2019 subsets)
* Constructs 5-class setup:

  * Depression
  * Anxiety
  * BPD
  * SuicideWatch
  * Control (merged non-mental-health subreddits)
* Cleans and balances the dataset
* Exports final processed CSV
* Implements classical baselines:

  * TF-IDF + Logistic Regression
  * TF-IDF + Linear SVM

---

### 2. `bert.ipynb`

* Trains BERT (`bert-base-uncased`) for multi-class classification
* Evaluates:

  * Accuracy
  * Macro F1
  * Confusion matrix
* Saves fine-tuned model for later reuse

---

### 3. `xai.ipynb`

* Loads trained BERT model
* Identifies Depression → SuicideWatch misclassifications
* Applies:

  * LIME
  * SHAP (pipeline-based)
* Analyzes token-level contributions driving crisis predictions
* Extracts recurring linguistic patterns in misclassified posts

---

## Key Findings (Current Progress)

* Classical linear models achieve ~0.76–0.77 macro F1.
* BERT improves macro F1 to ~0.80.
* Depression and SuicideWatch exhibit significant confusion.
* SHAP analysis shows predictions toward SuicideWatch are strongly driven by:

  * Explicit self-harm terminology
  * Existential finality language
  * Suicide-related vocabulary even in third-person contexts

---

## Technologies Used

* Python
* PyTorch
* HuggingFace Transformers
* Scikit-learn
* LIME
* SHAP
* Pandas / NumPy / Matplotlib / Seaborn

---

## Dataset

Reddit posts from mental health and control subreddits (2018–2019 subsets).
Subreddit membership is used as a proxy label for mental health condition.