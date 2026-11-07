
import re
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. REGEX-BASED TOKENIZER
# --------------------------------------------------

def tokenize_tweet(text):
    """
    Tokenize a tweet using regular expressions.

    Preserves URLs, mentions, hashtags, words,
    contractions, numbers, and punctuation.
    """

    if not isinstance(text, str):
        return []

    pattern = (
        r"https?://[^\s]+"
        r"|www\.[^\s]+"
        r"|@[A-Za-z0-9_]+"
        r"|#[A-Za-z0-9_]+"
        r"|[A-Za-z]+(?:['’][A-Za-z]+)*"
        r"|\d+(?:\.\d+)?"
        r"|[^\w\s]"
    )

    tokens = re.findall(pattern, text)

    return tokens


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

def load_dataset(file_path):

    column_names = [
        "tweet_id",
        "entity",
        "sentiment",
        "tweet_text"
    ]

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        header=None,
        names=column_names
    )

    print(f"\nDataset loaded: {file_path}")
    print(f"Total records: {len(df)}")

    return df


# --------------------------------------------------
# 3. APPLY TOKENIZATION
# --------------------------------------------------

def apply_tokenization(df):

    df = df.copy()

    df["tokens"] = df["tweet_text"].apply(tokenize_tweet)

    print("\nTokenization completed!")
    print(f"Total tweets: {len(df)}")

    print(
        "Tweets with tokens:",
        (df["tokens"].map(len) > 0).sum()
    )

    print(
        "Empty token lists:",
        (df["tokens"].map(len) == 0).sum()
    )

    return df


# --------------------------------------------------
# 4. SAVE TOKENIZED DATASET
# --------------------------------------------------

def save_dataset(df, output_path):

    df.to_csv(output_path, index=False)

    print(f"\nTokenized dataset saved: {output_path}")


# --------------------------------------------------
# 5. MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    # Find the main project directory
    project_dir = Path(__file__).resolve().parent.parent

    # Dataset paths
    train_path = project_dir / "data" / "train.csv"
    val_path = project_dir / "data" / "val.csv"

    # Output paths
    train_output = project_dir / "data" / "train_tokenized.csv"
    val_output = project_dir / "data" / "val_tokenized.csv"

    # Load datasets
    train_df = load_dataset(train_path)
    val_df = load_dataset(val_path)

    # Apply tokenization
    train_df = apply_tokenization(train_df)
    val_df = apply_tokenization(val_df)

    # Save tokenized datasets
    save_dataset(train_df, train_output)
    save_dataset(val_df, val_output)

    # Display an example
    print("\n========== TOKENIZATION EXAMPLE ==========")

    sample_tweet = train_df["tweet_text"].iloc[0]

    print("\nOriginal Tweet:")
    print(sample_tweet)

    print("\nTokenized Tweet:")
    print(train_df["tokens"].iloc[0])

    print("\n========== PROCESS COMPLETED ==========")
