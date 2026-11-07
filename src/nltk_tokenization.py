
import pandas as pd
from pathlib import Path
import sys
from nltk.tokenize import TweetTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import Regex tokenizer
from regex_tokenizaton import tokenize_tweet


# --------------------------------------------------
# 1. NLTK TOKENIZER
# --------------------------------------------------

nltk_tokenizer = TweetTokenizer(
    preserve_case=True,
    reduce_len=False,
    strip_handles=False
)


def tokenize_tweet_nltk(text):

    if not isinstance(text, str):
        return []

    return nltk_tokenizer.tokenize(text)


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
    print("Total records:", len(df))

    return df


# --------------------------------------------------
# 3. APPLY NLTK TOKENIZATION
# --------------------------------------------------

def apply_nltk_tokenization(df):

    df = df.copy()

    df["tokens"] = df["tweet_text"].apply(
        tokenize_tweet_nltk
    )

    print("\nNLTK tokenization completed!")

    return df


# --------------------------------------------------
# 4. COMPARE REGEX AND NLTK
# --------------------------------------------------

def compare_tokenization(df):

    df = df.copy()

    # Regex tokenization
    df["regex_tokens"] = df["tweet_text"].apply(
        tokenize_tweet
    )

    # NLTK tokenization
    df["nltk_tokens"] = df["tweet_text"].apply(
        tokenize_tweet_nltk
    )

    # Compare token lists
    df["same_output"] = (
        df["regex_tokens"] == df["nltk_tokens"]
    )

    df["regex_token_count"] = df["regex_tokens"].map(len)

    df["nltk_token_count"] = df["nltk_tokens"].map(len)

    return df


# --------------------------------------------------
# 5. DISPLAY COMPARISON RESULTS
# --------------------------------------------------

def display_comparison(df, dataset_name):

    total = len(df)

    identical = int(df["same_output"].sum())

    different = total - identical

    percentage = (
        (identical / total) * 100 if total > 0 else 0
    )

    print("\n======================================")
    print(f"{dataset_name} COMPARISON")
    print("======================================")

    print("Total tweets:", total)
    print("Identical outputs:", identical)
    print("Different outputs:", different)

    print(f"Matching percentage: {percentage:.2f}%")

    print(
        "Average Regex tokens:",
        round(df["regex_token_count"].mean(), 2)
    )

    print(
        "Average NLTK tokens:",
        round(df["nltk_token_count"].mean(), 2)
    )

    # Show examples of different results
    differences = df[df["same_output"] == False]

    print("\nExamples of different outputs:")

    if differences.empty:
        print("Both methods produced identical outputs.")

    else:
        for _, row in differences.head(5).iterrows():

            print("\nOriginal Tweet:")
            print(row["tweet_text"])

            print("Regex:", row["regex_tokens"])
            print("NLTK:", row["nltk_tokens"])


# --------------------------------------------------
# 6. MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    project_dir = Path(__file__).resolve().parent.parent

    train_path = project_dir / "data" / "train.csv"
    val_path = project_dir / "data" / "val.csv"

    # Load datasets
    train_df = load_dataset(train_path)
    val_df = load_dataset(val_path)

    # Apply NLTK tokenization
    train_nltk = apply_nltk_tokenization(train_df)
    val_nltk = apply_nltk_tokenization(val_df)

    # Save NLTK outputs
    train_nltk.to_csv(
        project_dir / "data" / "train_nltk_tokenized.csv",
        index=False
    )

    val_nltk.to_csv(
        project_dir / "data" / "val_nltk_tokenized.csv",
        index=False
    )

    # Compare both methods
    train_comparison = compare_tokenization(train_df)
    val_comparison = compare_tokenization(val_df)

    # Display comparison
    display_comparison(train_comparison, "TRAIN DATASET")
    display_comparison(val_comparison, "VALIDATION DATASET")

    # Save comparison files
    train_comparison.to_csv(
        project_dir / "data" / "train_tokenization_comparison.csv",
        index=False
    )

    val_comparison.to_csv(
        project_dir / "data" / "val_tokenization_comparison.csv",
        index=False
    )

    print("\n======================================")
    print("ALL OPERATIONS COMPLETED SUCCESSFULLY")
    print("======================================")
