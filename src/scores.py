import json
import os

SCORES_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scores.json")
MAX_SCORES = 10


def _is_valid_entry(entry) -> bool:
    """Voce di classifica valida: dict con name stringa e score intero."""
    if not isinstance(entry, dict):
        return False
    score = entry.get("score")
    return (isinstance(entry.get("name"), str)
            and isinstance(score, int) and not isinstance(score, bool))


def _read_scores() -> tuple[list[dict], bool]:
    """Legge scores.json in UTF-8. Ritorna (voci valide, file corrotto).

    File mancante, illeggibile o vuoto: ([], False). JSON non valido,
    encoding diverso o struttura diversa da una lista: ([], True). Le voci
    malformate di una lista vengono scartate.
    """
    try:
        with open(SCORES_FILE, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return [], False
    except ValueError:  # UnicodeDecodeError: file non in UTF-8
        return [], True
    if not text.strip():
        return [], False
    try:
        data = json.loads(text)
    except ValueError:  # JSONDecodeError (es. marcatori di un conflitto di merge)
        return [], True
    if not isinstance(data, list):
        return [], True
    return [s for s in data if _is_valid_entry(s)], False


def load_scores() -> list[dict]:
    """Carica la classifica da scores.json, ordinata per punteggio decrescente.

    Con file mancante, vuoto, non valido o con struttura inattesa ritorna
    una classifica vuota (o le sole voci valide): non solleva eccezioni.
    """
    scores, _ = _read_scores()
    scores.sort(key=lambda s: s["score"], reverse=True)
    return scores[:MAX_SCORES]


def save_score(name: str, score: int) -> int:
    """Salva un nuovo punteggio e ritorna la posizione in classifica (0-based).

    Ritorna -1 se il punteggio non rientra nella top 10 o se il file non
    si può scrivere (sola lettura, bloccato dalla sincronizzazione...).
    Se scores.json esiste ma è corrotto (es. conflitto di merge) viene
    prima rinominato in scores.json.bak, così non viene sovrascritto.
    La scrittura è atomica: file temporaneo nella stessa cartella e
    os.replace, quindi un'interruzione non lascia il file troncato.
    """
    scores, corrupted = _read_scores()
    if corrupted:
        try:
            os.replace(SCORES_FILE, SCORES_FILE + ".bak")
        except OSError:
            return -1  # meglio non salvare che perdere il file originale
    entry = {"name": name, "score": score}
    scores.append(entry)
    scores.sort(key=lambda s: s["score"], reverse=True)
    scores = scores[:MAX_SCORES]

    tmp = SCORES_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
        os.replace(tmp, SCORES_FILE)
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return -1

    # Posizione per identità: una voce uguale (stesso nome e score) già in
    # classifica non deve far risultare in top 10 una voce esclusa
    return next((i for i, s in enumerate(scores) if s is entry), -1)
