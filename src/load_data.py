import pandas as pd
import os

def load_datasets(data_dir="data/"):
    """
    Load both train and validation datasets from the local data directory.
    """
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError("Train or Validation CSV files not found in the 'data/' folder!")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    print(f"Train dataset loaded successfully! Rows: {len(train_df)}")
    print(f"Validation dataset loaded successfully! Rows: {len(val_df)}")
    
    return train_df, val_df

if __name__ == "__main__":
    train_df, val_df = load_datasets()