
import os
from gensim.models import Word2Vec, FastText


# 1. LOAD THE TRAINED MODELS


cbow_model = Word2Vec.load("models/cbow.model")
skipgram_model = Word2Vec.load("models/skipgram.model")
fasttext_model = FastText.load("models/fasttext.model")

models = {
    "CBOW": cbow_model,
    "Skip-gram": skipgram_model,
    "fastText": fasttext_model
}

print("All three embedding models loaded successfully!")



# 2. C1: TOP 5 MOST SIMILAR WORDS


# Candidate words related to sentiment and social media.
candidate_words = [
    "good", "bad", "love", "like", "game",
    "great", "hate", "play", "best", "worst"
]

# Select words available in all three vocabularies.
query_words = []

for word in candidate_words:
    if all(word in model.wv.key_to_index for model in models.values()):
        query_words.append(word)

query_words = query_words[:5]

print("\n========== C1: MOST SIMILAR WORDS ==========")

if len(query_words) < 3:
    print("Only these candidate words were found:", query_words)
    print("We may need to select other words from the dataset.")
else:
    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")

        for word in query_words:
            print(f"\nWords similar to '{word}':")

            similar_words = model.wv.most_similar(word, topn=5)

            for similar_word, score in similar_words:
                print(f"{similar_word:<20} {score:.4f}")



# 3. C2: COSINE SIMILARITY


similar_pairs = [
    ("good", "great"),
    ("bad", "worst")
]

unrelated_pairs = [
    ("love", "microsoft"),
    ("good", "xbox")
]

print("\n========== C2: COSINE SIMILARITY ==========")

for model_name, model in models.items():

    print(f"\n--- {model_name} ---")

    print("\nExpected Similar Word Pairs:")

    for word1, word2 in similar_pairs:
        if word1 in model.wv and word2 in model.wv:
            score = model.wv.similarity(word1, word2)
            print(f"{word1} - {word2}: {score:.4f}")
        else:
            print(f"{word1} - {word2}: Word not available")

    print("\nExpected Unrelated Word Pairs:")

    for word1, word2 in unrelated_pairs:
        if word1 in model.wv and word2 in model.wv:
            score = model.wv.similarity(word1, word2)
            print(f"{word1} - {word2}: {score:.4f}")
        else:
            print(f"{word1} - {word2}: Word not available")



# 4. C3: WORD ANALOGY


# Analogy: good - bad + terrible
# We attempt to find a word with a similar vector relationship.

print("\n========== C3: WORD ANALOGY ==========")

a = "good"
b = "bad"
c = "terrible"

for model_name, model in models.items():

    print(f"\n--- {model_name} ---")

    if all(word in model.wv for word in [a, b, c]):

        results = model.wv.most_similar(
            positive=[a, c],
            negative=[b],
            topn=5
        )

        print(f"Analogy: {a} - {b} + {c}")

        for word, score in results:
            print(f"{word:<20} {score:.4f}")

    else:
        print("One or more analogy words are not in the vocabulary.")

