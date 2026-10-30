"""Dati della dashboard HTML del training NEAT (dashboard/index.html).

Il trainer chiama record_generation() a fine generazione e update_live()
durante la valutazione. La pagina legge due script generati accanto a
index.html, caricati con tag <script> perche' funzioni aperta da file://
senza server:
- data.js: storico per generazione, dettaglio dell'ultima generazione e
  parametri del training (window.YS_DATA);
- live.js: fase e avanzamento della generazione in corso (window.YS_LIVE),
  con lo stamp di data.js per ricaricarlo solo quando cambia.
Lo storico persiste in una riga JSON per generazione (training_history.jsonl
in ai/), cosi' la dashboard mostra anche le sessioni precedenti.
Gli errori di scrittura non interrompono mai il training.
"""

import json
import math
import os
import time
import webbrowser
from pathlib import Path

LIVE_MIN_INTERVAL = 1.0  # secondi minimi tra due scritture di live.js


def _clean(obj, digits=3):
    """Arrotonda i float e sostituisce NaN/inf con None (JSON valido e compatto)."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, digits)
    if isinstance(obj, dict):
        return {str(k): _clean(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v, digits) for v in obj]
    return obj


def _write_js(path, var, payload):
    """Scrive `window.<var> = <json>;` in modo atomico (.tmp + os.replace)."""
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(f"window.{var} = ")
        json.dump(payload, f, ensure_ascii=False, separators=(',', ':'))
        f.write(";\n")
    os.replace(tmp, path)


class TrainingDashboard:
    """Storico e stato live del training, pubblicati per dashboard/index.html."""

    def __init__(self, static_info, history_path, out_dir):
        self.static = _clean(static_info)
        self.history_path = history_path
        self.out_dir = out_dir
        self.data_path = os.path.join(out_dir, 'data.js')
        self.live_path = os.path.join(out_dir, 'live.js')
        self.index_path = os.path.join(out_dir, 'index.html')
        self.session = int(time.time())
        self.history = self._load_history()
        self.snapshot = None
        self.stamp = 0
        self._last_live = 0.0
        self._warned = False

    def _load_history(self):
        """Righe valide dello storico (quelle illeggibili vengono saltate)."""
        rows = []
        try:
            with open(self.history_path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        continue
        except OSError:
            pass
        return rows

    def _warn(self, e):
        if not self._warned:
            print(f"Dashboard non aggiornata: {e!r}")
            self._warned = True

    def publish(self):
        """Riscrive data.js con storico, dettaglio e parametri correnti."""
        self.stamp = int(time.time() * 1000)
        payload = {
            'stamp': self.stamp,
            'static': self.static,
            'history': self.history,
            'snapshot': self.snapshot,
        }
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            _write_js(self.data_path, 'YS_DATA', payload)
        except OSError as e:
            self._warn(e)

    def record_generation(self, record, snapshot):
        """Aggiunge la generazione allo storico e ripubblica data.js."""
        record = _clean(dict(record, session=self.session))
        self.history.append(record)
        self.snapshot = _clean(snapshot)
        try:
            os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
            with open(self.history_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
        except OSError as e:
            self._warn(e)
        self.publish()

    def update_live(self, state, force=False):
        """Scrive live.js (al massimo una volta ogni LIVE_MIN_INTERVAL s, salvo force)."""
        now = time.time()
        if not force and now - self._last_live < LIVE_MIN_INTERVAL:
            return
        self._last_live = now
        payload = dict(state, updated=int(now * 1000), session=self.session,
                       data_stamp=self.stamp)
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            _write_js(self.live_path, 'YS_LIVE', _clean(payload))
        except OSError as e:
            self._warn(e)

    def open(self):
        """Apre la dashboard nel browser predefinito."""
        if os.path.exists(self.index_path):
            webbrowser.open(Path(self.index_path).as_uri())
