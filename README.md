# 📰 French Satire Detection with Machine Learning

This project tackles the challenge of **automatically detecting satirical vs. informative news headlines** in French. It leverages a pipeline of **custom web scraping**, **data cleaning**, and **machine learning**, culminating in a strong logistic regression baseline, with future extensions toward deep learning and LLMs.

---

## 🚀 Project Structure

- `data_collection.ipynb`  
  Scrapes headlines from 4 French news sources (2 satirical, 2 informative). Headlines are saved with source-based labels.

- `2_data_cleaning_and_merging.ipynb`  
  Applies cleaning rules to remove irrelevant, misleading, or malformed titles. Duplicates and non-French entries are also filtered out. Results are merged into a single dataset for modeling.

- `3_logistic_regression_baseline.ipynb`  
  Builds a TF-IDF + Logistic Regression baseline classifier. Includes detailed evaluation and interpretability (feature importance, confusion matrix, top tokens by class).

- `models/`  
  Contains the serialized final pipeline (`logistic_pipeline.joblib`).

- `data/`  
  Folder structure:
  - `1_raw/` contains unfiltered scraped data
  - `2_cleaned/` contains cleaned per-source datasets
  - `cleaned_dataset.csv` is the final ready-to-use dataset

---

## 📊 Current Dataset Summary

- **Raw data**: 10,164 headlines
- **Cleaned data**: 9,682 headlines  
  Balanced between satirical (Le Gorafi, Nordpresse) and informative (Le Figaro, Libération)

---

## 🧠 Current Model Performance

A logistic regression model trained on TF-IDF features achieves the following:

- **Accuracy**: ~82%
- **Precision/Recall (balanced)** across satire/informative classes
- **Clear interpretability** of the most discriminating tokens

---

## 🛠️ TODOs & Next Steps

### 📌 Data Cleaning Improvements

- [ ] **Outlier detection** for overly long titles  
      → visualize length distribution via mean + std / median + IQR  
- [ ] **Better token-level cleaning**: lowercase, stopword removal, tokenization, lemmatization (for traditional NLP models)

### 🧼 Refactoring Ideas

- [ ] Refactor `scrape()` for modularity  
      → isolate per-site logic (URL & parsing)
- [ ] Replace per-source CSVs in `data_collection` with raw `.txt` (1 title/line)  
      → only convert to CSV after cleaning + labeling

### 🧪 Modeling Enhancements

- [ ] Train a **Recurrent Neural Network (RNN/LSTM)** with word embeddings
- [ ] Fine-tune a **pre-trained French LLM** (e.g. CamemBERT, FlauBERT)  
      → Use HuggingFace transformers for tokenization + classification head

---

## 💡 Why This Project?

Satirical content is intentionally close to real news — making it both fascinating and challenging to detect. This project simulates a real-world scenario for text classification and is a great demonstration of:

- Applied **NLP pre-processing**
- **Feature engineering** & interpretability
- Multi-source **data wrangling**
- And soon, **deep learning model fine-tuning**

---

## 📎 Author

Created by [Your Name] — ML enthusiast with a passion for language and social context in AI.

---
