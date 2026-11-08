import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


from cleaning_core import core_clean_text


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class DatasetCleaningPipeline:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        
        
        self.stop_words = set(stopwords.words('english'))
        negations = {'not', 'no', 'never', 'none', 'nor', 'neither', 'ain', 'aren', "aren't", 
                     'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 
                     'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', 
                     "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 
                     'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}
        self.stop_words = self.stop_words - negations

    def advanced_clean(self, text):
        if not isinstance(text, str):
            return ""
       
        cleaned_base = core_clean_text(text)
        
      
        tokens = cleaned_base.split()
        final_tokens = []
        for token in tokens:
            if token not in self.stop_words:
                lemma = self.lemmatizer.lemmatize(token)
                final_tokens.append(lemma)
                
        return " ".join(final_tokens)

    def process_and_save(self, train_path, val_path):
        os.makedirs('outputs', exist_ok=True)
        
        for dataset_name, path in [('Train', train_path), ('Validation', val_path)]:
            if not os.path.exists(path):
                print(f"[{dataset_name}] File nahi mili yahan: {path}")
                continue
                
            print(f"Processing {dataset_name} dataset...")
            
            
            df = pd.read_csv(path, header=None)
            
          
            df.columns = ['id', 'entity', 'sentiment', 'text']
            
            
            raw_corpus = " ".join(df['text'].dropna().astype(str)).lower().split()
            vocab_before = set(raw_corpus)
            
            
            df['cleaned_text'] = df['text'].apply(self.advanced_clean)
            
            
            cleaned_corpus = " ".join(df['cleaned_text'].dropna().astype(str)).split()
            vocab_after = set(cleaned_corpus)
            
            print(f"--- Vocab Stats ({dataset_name}) ---")
            print(f"Vocab Size BEFORE: {len(vocab_before)}")
            print(f"Vocab Size AFTER:  {len(vocab_after)}")
            
            
            out_path = f"outputs/{dataset_name.lower()}_cleaned.csv"
            df.to_csv(out_path, index=False)
            print(f"Saved successfully to -> {out_path}\n")

if __name__ == "__main__":
    pipeline = DatasetCleaningPipeline()
    
    TRAIN_CSV = 'data/train.csv'
    VAL_CSV = 'data/val.csv'
    
    pipeline.process_and_save(TRAIN_CSV, VAL_CSV)