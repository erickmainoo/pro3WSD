# cs5322f25prog3.py

from typing import List
import os
import joblib


# -----------------------------
# Rule-based overrides
# -----------------------------

def _overtime_rule(sentence: str):
    """
    Rule-based override for 'overtime'.
    Return 1 or 2 if rule is confident, otherwise None.
    
    Sense 1: work done in addition to regular working hours
    Sense 2: extra game time beyond regulation, used to break a tie
    """
    s = sentence.lower()

    # Work / labor context
    work_words = [
        "work", "working", "hours", "shift", "shifts", "overtime shift",
        "pay", "paid", "unpaid", "wage", "wages", "salary", "rate",
        "employer", "employee", "staff", "union", "office", "job",
        "manager", "management", "boss", "supervisor",
        "compensation", "time bank", "holiday", "time-and-a-half",
        "double time", "overtime work", "timesheet", "schedule"
    ]

    # Sports / game context
    sport_words = [
        "game", "match", "team", "quarter", "period", "regulation",
        "shootout", "goal", "goals", "score", "scored", "scoring",
        "basketball", "football", "soccer", "hockey", "baseball",
        "playoffs", "postseason", "tournament", "finals",
        "win", "loss", "losing", "victory", "defeat",
        "ot", "pk", "penalty", "sudden death",
        "field", "court", "arena", "stadium",
        "coach", "head coach", "referee", "official"
    ]

    has_work = any(w in s for w in work_words)
    has_sport = any(w in s for w in sport_words)

    # Clearly work context only → sense 1
    if has_work and not has_sport:
        return 1

    # Clearly sports context only → sense 2
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
        # film/play/media context → sense 2
        return 2

    # Default: organization leader → sense 1
    return 1


def _rubbish_rule(sentence: str):
    """
    Rule-based override for 'rubbish'.

    Sense 1: physical trash/garbage (literal rubbish)
    Sense 2: 'nonsense' / 'low quality' / 'bad idea' (figurative)
    """
    s = sentence.lower()

    # Literal trash context
    trash_words = [
        "bin", "bins", "garbage", "trash", "waste", "litter",
        "landfill", "dump", "dumpster", "tip", "collection",
        "bag", "bags", "black bag", "refuse", "debris",
        "recycling", "rubbish truck", "garbage truck",
        "doorstep", "street", "pavement", "alley", "kerb", "curb"
    ]

    # Figurative "nonsense / bad" context
    nonsense_words = [
        "nonsense", "nonsensical", "ridiculous", "absurd",
        "idea", "argument", "theory", "policy", "proposal",
        "opinion", "excuse", "claim", "statement",
        "review", "critique", "analysis",
        "complete rubbish", "total rubbish",
        "absolute rubbish", "utter rubbish",
        "load of rubbish", "pile of rubbish"
    ]

    has_trash = any(w in s for w in trash_words)
    has_nonsense = any(w in s for w in nonsense_words)

    if has_trash and not has_nonsense:
        # literal garbage → sense 1
        return 1

    if has_nonsense and not has_trash:
        # figurative nonsense → sense 2
        return 2

    # Ambiguous or neutral: let ML decide
    return None


# -----------------------------
# Model loading and prediction
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def _load_model_for_word(word: str):
    """
    Load TF-IDF vectorizer and classifier for the given word.
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

    Pipeline:
    1. Try rule-based override for the word.
    2. If rule is not confident (returns None), fall back to the ML classifier.
    """
    if not sentences:
        return []

    vectorizer, clf = _load_model_for_word(word)
    final_preds: List[int] = []

    for sent in sentences:
        # 1) Rule-based override
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

        # 2) ML fallback
        processed = _preprocess_for_word([sent], word)
        X_vec = vectorizer.transform(processed)
        model_pred = clf.predict(X_vec)[0]
        final_preds.append(int(model_pred))

    return final_preds


# -----------------------------
# Public API (required by prof)
# -----------------------------

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

