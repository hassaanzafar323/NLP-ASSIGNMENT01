
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# SETUP & DATA LOADING
train_path = 'outputs/train_cleaned.csv'
if os.path.exists(train_path):
    df = pd.read_csv(train_path)
    corpus = df['cleaned_text'].dropna().astype(str).tolist()
else:
    df = pd.read_csv('data/train.csv', header=None)
    df.columns = ['id', 'entity', 'sentiment', 'text']
    corpus = df['text'].dropna().astype(str).tolist()

print(f"Total documents loaded: {len(corpus)}\n")



# A1. BAG OF WORDS (BoW)
count_vec = CountVectorizer(max_features=5000)
bow_matrix = count_vec.fit_transform(corpus)

bow_rows, bow_cols = bow_matrix.shape
bow_total = bow_rows * bow_cols
bow_sparsity = ((bow_total - bow_matrix.nnz) / bow_total) * 100

print("=== A1. Bag of Words (BoW) ===")
print(f"Matrix Shape: {bow_matrix.shape} (Docs x Features)")
print(f"Sparsity: {bow_sparsity:.2f}% zero entries\n")



# A2. TF-IDF REPRESENTATION

tfidf_vec = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vec.fit_transform(corpus)
feature_names = tfidf_vec.get_feature_names_out()

print("=== A2. TF-IDF Representation ===")
print(f"Matrix Shape: {tfidf_matrix.shape}")

# Sample high-weight words analysis from the first document
sample_row = tfidf_matrix.getrow(0).toarray().flatten()
top_idx = sample_row.argsort()[::-1][:3]
print("Top high-weight words in sample doc:")
for idx in top_idx:
    if sample_row[idx] > 0:
        print(f"  - '{feature_names[idx]}': {sample_row[idx]:.4f}")
print()



# A3. N-GRAMS (Unigrams, Bigrams, Trigrams)

ngram_vec = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
ngram_matrix = ngram_vec.fit_transform(corpus)

n_rows, n_cols = ngram_matrix.shape
n_total = n_rows * n_cols
n_sparsity = ((n_total - ngram_matrix.nnz) / n_total) * 100

print("=== A3. N-grams (1 to 3) ===")
print(f"Vocabulary Size (Feature Count): {n_cols}")
print(f"Matrix Sparsity with N-grams: {n_sparsity:.2f}%")