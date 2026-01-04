# Property Address Classification

## Project Overview
This project is an AI/ML solution designed to classify raw property address text into predefined categories. [cite_start]The goal was to build a robust classifier capable of understanding noisy address data and assigning it to one of five classes: `flat`, `houseorplot`, `landparcel`, `commercial unit`, or `others`.

This repository contains the training of the model, exploratory data analysis, and the final model implementation using **DistilBERT**, which achieved **93% accuracy** on the validation dataset.

---

## Dataset Details
The model was trained on a provided dataset consisting of raw text strings of property addresses[cite: 10].
- **Input:** `property_address` (Raw text)
- **Target:** `categories` (Categorical label)
- **Classes:**
  - `flat`
  - `houseorplot`
  - `landparcel`
  - `commercial unit`
  - `others`

## Approach & Methodology

I adopted an iterative approach, starting with a strong Machine Learning baseline and then moving to a Deep Learning solution to maximize performance.

### 1. Baseline: Logistic Regression
- **Preprocessing:** Text cleaning (lowercasing, removing special characters).
- **Feature Extraction:** **TF-IDF (Term Frequency-Inverse Document Frequency)** with bigrams (`ngram_range=(1,2)`) to capture context like "Plot No" or "Shop No".
- **Model:** Logistic Regression with `class_weight='balanced'`.
- **Result:** ~87% Accuracy.

### 2. Final Model: DistilBERT (Transfer Learning)
- **Model:** Fine-tuned **`distilbert-base-uncased`**.
- **Configuration:**
  - **Epochs:** 3
  - **Batch Size:** 16
  - **Learning Rate:** 2e-5
- **Result:** **92% Accuracy**