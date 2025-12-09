# cs5322f25prog3.py

from typing import List
import os
import joblib


def _overtime_rule(sentence: str):
    """
    Rule-based override for 'overtime'.
    Return 1 or 2 if rule is confident, otherwise None.
    
    Sense 1: work done in addition to regular working hours
    Sense 2: extra game time / sports context
    """
    s = sentence.lower()

    work_words = [
        "work", "working", "hours", "shift", "shifts", "pay",
        "paid", "unpaid", "wage", "wages", "salary", "employer",
        "union", "office", "job", "manager", "management",
        "compensation", "time bank", "holiday", "overtime work",
        "collective agreement"
    ]
    sport_words = [
        "game", "match", "team", "quarter", "period", "regulation",
        "shootout", "goal", "goals", "score", "scored",
        "basketball", "football", "hockey", "soccer",
        "win", "loss", "losing", "playoffs", "ot", "pk", "penalty",
        "field", "court", "coach", "head coach", "ncaa"
    ]

    has_work = any(w in s for w in work_words)
    has_sport = any(w in s for w in sport_words)

    # If clearly work context and not sport → sense 1
    if has_work and not has_sport:
        return 1
    # If clearly sport context and not work → sense 2
    if has_sport and not has_work:
        return 2

    # Otherwise, let the ML model decide
    return None


def _director_rule(sentence: str):
    """
    Rule-based override for 'director'.

    Sense 1: person who leads an organization (company, institution, etc.)
    Sense 2: person in charge of making a film/play.
    """
    s = sentence.lower()

    film_words = [
        "film", "movie", "cinema", "theatre", "theater", "play",
        "tv", "television", "documentary", "producer", "cast",
        "actor", "actress", "shot", "scene", "script", "screenplay",
        "hollywood", "oscar", "festival", "nominated",
        "choreography", "special effects"
    ]

    has_film = any(w in s for w in film_words)

    if has_film:
        return 2  # film/play director (sense 2)

    # Default: organization leader (sense 1)
    return 1


def _rubbish_rule(sentence: str):
    """
    Rule-based override for 'rubbish'.
    Sense 1: physical trash/garbage
    Sense 2: 'nonsense' / low quality / bad idea
    """
    s = sentence.lower()

    trash_words = [
        "bin", "bins", "garbage", "trash", "waste", "litter",
        "landfill", "dump", "collection", "bag", "bags",
        "recycling", "doorstep", "street", "pavement"
    ]
    nonsense_words = [
        "nonsense", "idea", "argument", "theory", "policy",
        "proposal", "opinion", "complete rubbish", "total rubbish",
        "absolute rubbish", "utter rubbish", "excuse", "statement",
        "claim", "review", "film", "movie"
    ]

    has_trash = any(w in s for w in trash_words)
    has_nonsense = any(w in s for w in nonsense_words)

    if has_trash and not has_nonsense:
        return 1  # physical trash
    if has_nonsense and not has_trash:
        return 2  # nonsense / criticism

    return None


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
    Uses rule-based overrides first, then ML model as fallback.
    For 'director', rules are strong enough to always decide.
    """
    if not sentences:
        return []

    vectorizer, clf = _load_model_for_word(word)

    final_preds: List[int] = []

    for sent in sentences:
        # 1) Try rule-based override
        rule_label = None
        if word == "overtime":
            rule_label = _overtime_rule(sent)
        elif word == "director":
            rule_label = _director_rule(sent)
        elif word == "rubbish":
            rule_label = _rubbish_rule(sent)

        if rule_label in (1, 2):
            final_preds.append(rule_label)
            continue

        # 2) Fall back to model
        processed = _preprocess_for_word([sent], word)
        X_vec = vectorizer.transform(processed)
        model_pred = clf.predict(X_vec)[0]
        final_preds.append(int(model_pred))

    return final_preds


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
