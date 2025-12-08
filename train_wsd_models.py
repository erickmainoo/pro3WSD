# train_wsd_models.py

import os
from typing import List, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Directories (you can adjust these if needed)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


def load_wsd_file(path: str):
    """
    Handles original dataset format:
    
    Word Title (ignored)
    1 <gloss for sense 1>
    2 <gloss for sense 2>

    1        <-- start of sense 1 sentences
    sentence
    sentence
    sentence

    2        <-- start of sense 2 sentences
    sentence
    sentence
    ...
    """

    sentences = []
    labels = []

    current_label = None
    in_sentence_section = False

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue  # skip blank lines

            # Detect section change for actual training sentences
            if line == "1":
                current_label = 1
                in_sentence_section = True
                continue
            if line == "2":
                current_label = 2
                in_sentence_section = True
                continue

            # Ignore gloss lines (format: "1 <text>" or "2 <text>")
            if line.startswith("1 ") or line.startswith("2 "):
                # This line is gloss part -- skip
                continue

            # If we are inside actual sentence section, add labeled sentence
            if in_sentence_section and current_label in (1, 2):
                sentences.append(line)
                labels.append(current_label)

    return sentences, labels



def preprocess_sentences(sentences: List[str], target_word: str) -> List[str]:
    """
    Simple preprocessing:
    - Lowercase
    - Remove the target word itself (singular & plural) from the sentence.
    """
    word = target_word.lower()
    plural = word + "s"

    processed = []
    for s in sentences:
        s_lower = s.lower()
        tokens = s_lower.split()
        tokens = [t for t in tokens if t != word and t != plural]
        processed.append(" ".join(tokens))
    return processed


def train_one_word(
    word: str,
    train_filename: str,
    ngram_range=(1, 2),
    C: float = 1.0,
):
    """
    Train a WSD model for a single word and save:
        models/<word>_vectorizer.joblib
        models/<word>_model.joblib
    """
    path = os.path.join(DATA_DIR, train_filename)
    print(f"Training for '{word}' from file: {path}")

    sentences, labels = load_wsd_file(path)
    print(f"  Loaded {len(sentences)} sentences, labels: {set(labels)}")

    X_proc = preprocess_sentences(sentences, word)

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=1,
        max_df=0.95,
    )
    X_vec = vectorizer.fit_transform(X_proc)

    # Logistic Regression classifier
    clf = LogisticRegression(max_iter=1000, C=C)
    clf.fit(X_vec, labels)

    # Save model & vectorizer
    vec_path = os.path.join(MODEL_DIR, f"{word}_vectorizer.joblib")
    model_path = os.path.join(MODEL_DIR, f"{word}_model.joblib")

    joblib.dump(vectorizer, vec_path)
    joblib.dump(clf, model_path)

    print(f"  Saved vectorizer -> {vec_path}")
    print(f"  Saved model      -> {model_path}")
    print()


def main():
    # Adjust filenames here if your prof used slightly different names.
    train_one_word("director", "director.txt")
    train_one_word("overtime", "overtime.txt")
    train_one_word("rubbish", "rubbish.txt")

    print("All models trained and saved in 'models/'.")


if __name__ == "__main__":
    main()
