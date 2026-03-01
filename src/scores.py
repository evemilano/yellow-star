import json
import os

SCORES_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scores.json")
MAX_SCORES = 10


def load_scores() -> list[dict]:
    """Carica la classifica da scores.json, ordinata per punteggio decrescente."""
    try:
        with open(SCORES_FILE, "r", encoding="utf-8") as f:
            scores = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    scores.sort(key=lambda s: s["score"], reverse=True)
    return scores[:MAX_SCORES]


def save_score(name: str, score: int) -> int:
    """Salva un nuovo punteggio e ritorna la posizione in classifica (0-based).

    Ritorna -1 se il punteggio non rientra nella top 10.
    """
    scores = load_scores()
    entry = {"name": name, "score": score}
    scores.append(entry)
    scores.sort(key=lambda s: s["score"], reverse=True)
    scores = scores[:MAX_SCORES]

    with open(SCORES_FILE, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    try:
        rank = scores.index(entry)
    except ValueError:
        return -1
    return rank
