import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, FastText

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# -------------------------------------------------------------------------
# D1: Save each representation/model to disk
# -------------------------------------------------------------------------
def save_tfidf_vectorizer():
    print("Loading cleaned dataset for vectorizer fitting...")
    if os.path.exists("outputs/train_cleaned.csv"):
        df = pd.read_csv("outputs/train_cleaned.csv")
        
        # FIX: NaN values ko safe handle karne ke liye fillna("") use kiya hai
        X_text = df['text'].fillna("").astype(str)
        
        print("Training and saving TF-IDF vectorizer...")
        tfidf = TfidfVectorizer(max_features=5000)
        tfidf.fit(X_text)
        joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
        print("Saved: models/tfidf_vectorizer.pkl")
    else:
        print("Warning: 'outputs/train_cleaned.csv' not found. Ensure dataset is available.")

# -------------------------------------------------------------------------
# D2: Unified Inference Interface for Phase 3
# -------------------------------------------------------------------------
def get_document_vector(raw_text: str, representation: str = "cbow") -> np.ndarray:
    """
    Unified inference function for Phase 3.
    Given a raw document and representation type, returns its vector.
    
    Supported representations: 'tfidf', 'cbow', 'skipgram', 'fasttext'
    """
    representation = representation.lower()
    
    if representation == "tfidf":
        vectorizer_path = "models/tfidf_vectorizer.pkl"
        if not os.path.exists(vectorizer_path):
            save_tfidf_vectorizer()
        vectorizer = joblib.load(vectorizer_path)
        return vectorizer.transform([raw_text]).toarray()[0]
        
    elif representation in ["cbow", "skipgram", "fasttext"]:
        model_path = f"models/{representation}.model"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found. Ensure it is trained and saved.")
            
        model = Word2Vec.load(model_path) if representation != "fasttext" else FastText.load(model_path)
        tokens = raw_text.split()
        vector_size = model.vector_size
        
        vectors = [model.wv[word] for word in tokens if word in model.wv]
        if len(vectors) == 0:
            return np.zeros(vector_size)
        return np.mean(vectors, axis=0)
        
    else:
        raise ValueError(f"Unknown representation type: '{representation}'. Use 'tfidf', 'cbow', 'skipgram', or 'fasttext'.")


if __name__ == "__main__":
    # Ensure TF-IDF vectorizer is saved if not already present
    if not os.path.exists("models/tfidf_vectorizer.pkl"):
        save_tfidf_vectorizer()
        
    sample_text = "this is an amazing and wonderful product experience"
    
    print("\n--- Testing Phase 3 Inference Interface (D2) ---")
    for rep in ["tfidf", "cbow", "skipgram", "fasttext"]:
        try:
            vec = get_document_vector(sample_text, representation=rep)
            print(f"[{rep.upper()}] Vector shape: {vec.shape}, Non-zero elements: {np.count_nonzero(vec)}")
        except Exception as e:
            print(f"[{rep.upper()}] Skipped: {e}")