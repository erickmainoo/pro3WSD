# cs5322f25prog3.py

from typing import List
import os
import joblib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def _load_model_for_word(word: str):
    """
    Load vectorizer and classifier for the given word.
    """
    vec_path = os.path.join(MODEL_DIR, f"{word}_vectorizer.joblib")
    model_path = os.path.join(MODEL_DIR, f"{word}_model.joblib")

    if not os.path.exists(vec_path):
        raise FileNotFoundError(f"Vectorizer file not found: {vec_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    vectorizer = joblib.load(vec_path)
    clf = joblib.load(model_path)
    return vectorizer, clf


def _preprocess_for_word(sentences: List[str], word: str) -> List[str]:
    """
    Same preprocessing as in train_wsd_models.py:
    - lowercase
    - remove the target word (singular / plural)
    """
    word = word.lower()
    plural = word + "s"

    processed: List[str] = []
    for s in sentences:
        s_lower = s.lower()
        tokens = s_lower.split()
        tokens = [t for t in tokens if t != word and t != plural]
        processed.append(" ".join(tokens))
    return processed


def _predict_word(word: str, sentences: List[str]) -> List[int]:
    """
    Generic prediction helper for a single word.
    """
    if not sentences:
        return []

    vectorizer, clf = _load_model_for_word(word)
    processed = _preprocess_for_word(sentences, word)
    X_vec = vectorizer.transform(processed)
    preds = clf.predict(X_vec)

    # Ensure plain Python ints (1 or 2)
    return [int(p) for p in preds]


def WSD_Test_director(sent_list: List[str]) -> List[int]:
    """
    Input: list of sentences containing the word 'director'
    Output: list of sense IDs (1 or 2)
    """
    return _predict_word("director", sent_list)


def WSD_Test_rubbish(sent_list: List[str]) -> List[int]:
    """
    Input: list of sentences containing the word 'rubbish'
    Output: list of sense IDs (1 or 2)
    """
    return _predict_word("rubbish", sent_list)


def WSD_Test_overtime(sent_list: List[str]) -> List[int]:
    """
    Input: list of sentences containing the word 'overtime'
    Output: list of sense IDs (1 or 2)
    """
    return _predict_word("overtime", sent_list)
