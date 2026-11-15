import os
import re
import pandas as pd
from gensim.models import Word2Vec, FastText

# ==========================================
# 1. Data Loading & Tokenization
# ==========================================
def load_and_tokenize(file_path):
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path, header=None)
    texts = df[3].dropna().astype(str).tolist()
    print(f"Total documents loaded: {len(texts)}")

    tokenized_corpus = []
    for text in texts:
        clean_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        tokens = clean_text.split()
        if tokens:
            tokenized_corpus.append(tokens)

    return tokenized_corpus

# ==========================================
# 2. Train Word Embeddings
# ==========================================
def train_embeddings(sentences):
    VECTOR_SIZE = 100
    WINDOW = 5
    MIN_COUNT = 2
    EPOCHS = 10
    NEGATIVE = 5
    WORKERS = 4

    print("\n--- Training CBOW Model (sg=0) ---")
    cbow_model = Word2Vec(
        sentences=sentences, 
        vector_size=VECTOR_SIZE, 
        window=WINDOW, 
        min_count=MIN_COUNT, 
        sg=0, 
        negative=NEGATIVE, 
        epochs=EPOCHS, 
        workers=WORKERS, 
        seed=42
    )
    print(f"CBOW Vocabulary Size: {len(cbow_model.wv)}")

    print("\n--- Training Skip-gram Model (sg=1) ---")
    skipgram_model = Word2Vec(
        sentences=sentences, 
        vector_size=VECTOR_SIZE, 
        window=WINDOW, 
        min_count=MIN_COUNT, 
        sg=1, 
        negative=NEGATIVE, 
        epochs=EPOCHS, 
        workers=WORKERS, 
        seed=42
    )
    print(f"Skip-gram Vocabulary Size: {len(skipgram_model.wv)}")

    print("\n--- Training fastText Model ---")
    fasttext_model = FastText(
        sentences=sentences, 
        vector_size=VECTOR_SIZE, 
        window=WINDOW, 
        min_count=MIN_COUNT, 
        sg=1, 
        min_n=3, 
        max_n=6, 
        negative=NEGATIVE, 
        epochs=EPOCHS, 
        workers=WORKERS, 
        seed=42
    )
    print(f"fastText Vocabulary Size: {len(fasttext_model.wv)}")

    return cbow_model, skipgram_model, fasttext_model

# ==========================================
# 3. Dynamic OOV & Misspelled Word Evaluation
# ==========================================
def evaluate_models(skipgram, fasttext):
    vocab = list(skipgram.wv.key_to_index.keys())
    
    # Pick a real word from dataset vocabulary dynamically
    base_word = "murder" if "murder" in vocab else vocab[100]
    
    # Introduce an intentional typo to create an OOV/misspelled test token
    if len(base_word) > 4:
        misspelled_word = base_word[:-2] + "cr" # e.g., 'murder' -> 'murdcr'
    else:
        misspelled_word = base_word + "xyz"

    print(f"\n==========================================")
    print(f"Dynamic OOV / Misspelled Word Demonstration")
    print(f"==========================================")
    print(f"Base Word from Dataset: '{base_word}'")
    print(f"Generated Misspelled/OOV Test Token: '{misspelled_word}'")

    # 1. Word2Vec (Skip-gram) check
    print(f"\n[Word2Vec Skip-gram Check]:")
    if misspelled_word in skipgram.wv:
        print(f"  Found unexpectedly: {skipgram.wv.most_similar(misspelled_word, topn=3)}")
    else:
        print(f"  -> KeyError: '{misspelled_word}' is NOT in Word2Vec vocabulary (OOV limitation).")

    # 2. fastText subword resolution check
    print(f"\n[fastText Subword Resolution Check]:")
    try:
        similar_ft = fasttext.wv.most_similar(misspelled_word, topn=3)
        formatted_ft = [(w, round(s, 4)) for w, s in similar_ft]
        print(f"  -> Success! fastText successfully computed subword vector for '{misspelled_word}':")
        for w, s in formatted_ft:
            print(f"     {w:<15} : {s}")
    except Exception as e:
        print(f"  -> Error: {e}")

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    DATA_PATH = "outputs/train_cleaned.csv" 
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: File not found at '{DATA_PATH}'. Please update DATA_PATH.")
    else:
        corpus = load_and_tokenize(DATA_PATH)
        cbow_model, sg_model, ft_model = train_embeddings(corpus)
        evaluate_models(sg_model, ft_model)
        
        # Save models for Part D
        os.makedirs("models", exist_ok=True)
        cbow_model.save("models/cbow.model")
        sg_model.save("models/skipgram.model")
        ft_model.save("models/fasttext.model")
        print("\nAll Part B models successfully saved in 'models/' directory!")