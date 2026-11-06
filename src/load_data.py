import os
import pandas as pd

def load_and_explore_data():
    
    train_path = os.path.join("data", "train.csv")
    val_path = os.path.join("data", "val.csv")
    
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("Error: train.csv ya val.csv 'data/' folder ke andar nahi mili!")
        return None, None
    
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
   
    print(" TASK 1 & 2: DATASET EXPLORATION REPORT ")
    
    
    print(f"-> Training Documents (Rows): {train_df.shape[0]}")
    print(f"-> Validation Documents (Rows): {val_df.shape[0]}")
    
    
    print("\nColumns found in dataset:", list(train_df.columns))
    
    text_col = train_df.columns[0]  
    label_col = train_df.columns[1] if len(train_df.columns) > 1 else None
    
    
    if text_col:
        train_df['word_count'] = train_df[text_col].astype(str).apply(lambda x: len(x.split()))
        avg_len = train_df['word_count'].mean()
        print(f"-> Average Document Length: {avg_len:.2f} words")
    
   
    if label_col:
        print("\n-> Target Distribution (Training Set):")
        print(train_df[label_col].value_counts())
        
    
    print("\n-> Examples of Raw Text (First 3 rows):")
    for i in range(min(3, len(train_df))):
        print(f"  [{i+1}] {train_df[text_col].iloc[i]}")
        
   
    return train_df, val_df

if __name__ == "__main__":
    load_and_explore_data()