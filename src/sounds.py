"""Sistema audio procedurale — tutti i suoni generati via codice."""

import math
import struct
import random
import pygame


# ── Costanti audio ──
SAMPLE_RATE = 44100
MAX_16BIT = 32767


def _make_sound(samples: list[int]) -> pygame.mixer.Sound:
    """Crea un pygame.mixer.Sound da una lista di campioni signed-16bit.

    Duplica ogni campione su L+R perché SDL forza stereo.
    """
    stereo = []
    for s in samples:
        stereo.append(s)
        stereo.append(s)
    raw = struct.pack(f"<{len(stereo)}h", *stereo)
    return pygame.mixer.Sound(buffer=raw)


def _clamp(v: float) -> int:
    return max(-MAX_16BIT, min(MAX_16BIT, int(v)))


# ═══════════════════════════════════════════════════════════
#  Generazione effetti sonori
# ═══════════════════════════════════════════════════════════

def _gen_shoot() -> pygame.mixer.Sound:
    """Sparo laser corto e incisivo."""
    duration = 0.12
    n = int(SAMPLE_RATE * duration)
    samples = []
    for i in range(n):
        t = i / SAMPLE_RATE
        env = max(0.0, 1.0 - t / duration)  # fade out lineare
        # Frequenza che scende rapidamente (da 1200 a 400 Hz)
        freq = 1200 - 6000 * t
        val = math.sin(2 * math.pi * freq * t) * 0.5
        # Aggiunge un po' di quadra per crunch
        sq = 1.0 if math.sin(2 * math.pi * freq * 1.5 * t) > 0 else -1.0
        val = val * 0.7 + sq * 0.3
        samples.append(_clamp(val * env * MAX_16BIT * 0.35))
    return _make_sound(samples)


def _gen_explosion() -> pygame.mixer.Sound:
    """Esplosione: rumore filtrato + bassa frequenza."""
    duration = 0.45
    n = int(SAMPLE_RATE * duration)
    samples = []
    prev = 0.0
    for i in range(n):
        t = i / SAMPLE_RATE
        # Envelope: attacco rapido, decay lento
        if t < 0.02:
            env = t / 0.02
        else:
            env = max(0.0, 1.0 - (t - 0.02) / (duration - 0.02))
        env = env ** 1.5  # curva più morbida

        # Rumore filtrato (rumore bianco + filtro passa-basso semplice)
        noise = random.uniform(-1.0, 1.0)
        alpha = 0.15 + 0.3 * (1.0 - t / duration)  # filtro si chiude nel tempo
        prev = prev * (1.0 - alpha) + noise * alpha
        val = prev

        # Sub-bass boom (60 Hz)
        boom = math.sin(2 * math.pi * 60 * t) * 0.6
        # Tono medio che decade (200 Hz → 80 Hz)
        mid_freq = 200 - 260 * t
        mid = math.sin(2 * math.pi * mid_freq * t) * 0.3

        val = val * 0.5 + boom * env + mid * env
        samples.append(_clamp(val * env * MAX_16BIT * 0.5))
    return _make_sound(samples)


def _gen_powerup() -> pygame.mixer.Sound:
    """Suono raccolta power-up: arpeggio ascendente luminoso."""
    duration = 0.35
    n = int(SAMPLE_RATE * duration)
    samples = []
    # 4 note rapide ascendenti
    notes = [523, 659, 784, 1047]  # Do5, Mi5, Sol5, Do6
    note_dur = duration / len(notes)
    for i in range(n):
        t = i / SAMPLE_RATE
        note_idx = min(int(t / note_dur), len(notes) - 1)
        freq = notes[note_idx]
        local_t = t - note_idx * note_dur
        # Envelope per nota singola
        env = max(0.0, 1.0 - local_t / note_dur)
        # Onda sinusoidale + armoniche leggere
        val = math.sin(2 * math.pi * freq * t) * 0.6
        val += math.sin(2 * math.pi * freq * 2 * t) * 0.25
        val += math.sin(2 * math.pi * freq * 3 * t) * 0.1
        # Envelope globale (fade in rapido)
        g_env = min(1.0, t / 0.01)
        samples.append(_clamp(val * env * g_env * MAX_16BIT * 0.3))
    return _make_sound(samples)


def _gen_player_hit() -> pygame.mixer.Sound:
    """Colpo subito dal giocatore: impatto + tono discendente."""
    duration = 0.3
    n = int(SAMPLE_RATE * duration)
    samples = []
    prev = 0.0
    for i in range(n):
        t = i / SAMPLE_RATE
        env = max(0.0, 1.0 - t / duration) ** 1.2
        # Impatto iniziale (rumore)
        noise = random.uniform(-1.0, 1.0)
        alpha = 0.3 if t < 0.05 else 0.08
        prev = prev * (1.0 - alpha) + noise * alpha
        # Tono che scende (400 → 100 Hz)
        freq = 400 - 1000 * t
        tone = math.sin(2 * math.pi * freq * t) * 0.5
        val = prev * 0.4 + tone * 0.6
        samples.append(_clamp(val * env * MAX_16BIT * 0.4))
    return _make_sound(samples)


def _gen_game_over() -> pygame.mixer.Sound:
    """Game over: triade discendente triste."""
    duration = 1.2
    n = int(SAMPLE_RATE * duration)
    samples = []
    # Note discendenti (Mi4, Do4, La3) con pause
    notes = [(330, 0.0, 0.35), (262, 0.38, 0.35), (220, 0.76, 0.42)]
    for i in range(n):
        t = i / SAMPLE_RATE
        val = 0.0
        for freq, start, dur in notes:
            if start <= t < start + dur:
                local_t = t - start
                # Envelope con sustain
                if local_t < 0.02:
                    env = local_t / 0.02
                elif local_t > dur - 0.08:
                    env = (dur - local_t) / 0.08
                else:
                    env = 1.0
                env = max(0.0, env) * 0.85
                # Onda triangolare (più dolce della quadra)
                phase = (freq * t) % 1.0
                wave = 2.0 * abs(2.0 * phase - 1.0) - 1.0
                # Aggiunge sinusoide per calore
                wave = wave * 0.4 + math.sin(2 * math.pi * freq * t) * 0.6
                val += wave * env
        samples.append(_clamp(val * MAX_16BIT * 0.3))
    return _make_sound(samples)


def _gen_missile_launch() -> pygame.mixer.Sound:
    """Lancio missile: whoosh ascendente."""
    duration = 0.25
    n = int(SAMPLE_RATE * duration)
    samples = []
    prev = 0.0
    for i in range(n):
        t = i / SAMPLE_RATE
        env = max(0.0, 1.0 - t / duration)
        # Frequenza che sale
        freq = 200 + 3000 * t
        tone = math.sin(2 * math.pi * freq * t) * 0.4
        # Rumore filtrato (jet)
        noise = random.uniform(-1.0, 1.0)
        alpha = 0.2 + 0.3 * (t / duration)
        prev = prev * (1.0 - alpha) + noise * alpha
        val = tone + prev * 0.5
        samples.append(_clamp(val * env * MAX_16BIT * 0.3))
    return _make_sound(samples)


def _gen_bomb() -> pygame.mixer.Sound:
    """Bomba: esplosione grossa + onda d'urto."""
    duration = 0.8
    n = int(SAMPLE_RATE * duration)
    samples = []
    prev = 0.0
    for i in range(n):
        t = i / SAMPLE_RATE
        # Envelope con attacco istantaneo
        if t < 0.01:
            env = t / 0.01
        else:
            env = max(0.0, 1.0 - (t - 0.01) / (duration - 0.01)) ** 1.3
        # Sub-bass potente (40 Hz)
        boom = math.sin(2 * math.pi * 40 * t) * 0.8
        # Secondo basso (80 Hz)
        boom2 = math.sin(2 * math.pi * 80 * t) * 0.4
        # Rumore crunch
        noise = random.uniform(-1.0, 1.0)
        alpha = 0.25 if t < 0.1 else 0.1
        prev = prev * (1.0 - alpha) + noise * alpha
        val = boom + boom2 + prev * 0.5
        samples.append(_clamp(val * env * MAX_16BIT * 0.45))
    return _make_sound(samples)


def _gen_shield_hit() -> pygame.mixer.Sound:
    """Colpo assorbito dallo scudo: tono cristallino metallico."""
    duration = 0.2
    n = int(SAMPLE_RATE * duration)
    samples = []
    for i in range(n):
        t = i / SAMPLE_RATE
        env = max(0.0, 1.0 - t / duration) ** 1.5
        # Tono alto metallico (800 Hz + armonica)
        val = math.sin(2 * math.pi * 800 * t) * 0.4
        val += math.sin(2 * math.pi * 1600 * t) * 0.25
        val += math.sin(2 * math.pi * 2400 * t) * 0.15
        # Leggero "ting" iniziale
        if t < 0.03:
            val += math.sin(2 * math.pi * 3200 * t) * 0.3 * (1.0 - t / 0.03)
        samples.append(_clamp(val * env * MAX_16BIT * 0.35))
    return _make_sound(samples)


def _gen_shield_break() -> pygame.mixer.Sound:
    """Rottura scudo: vetro che si frantuma + tono discendente."""
    duration = 0.5
    n = int(SAMPLE_RATE * duration)
    samples = []
    prev = 0.0
    for i in range(n):
        t = i / SAMPLE_RATE
        # Envelope con attacco forte e decay
        if t < 0.015:
            env = t / 0.015
        else:
            env = max(0.0, 1.0 - (t - 0.015) / (duration - 0.015)) ** 1.3
        # Rumore tipo vetro (passa-alto)
        noise = random.uniform(-1.0, 1.0)
        alpha = 0.5 if t < 0.08 else 0.2
        prev = prev * (1.0 - alpha) + noise * alpha
        # Tono discendente cristallino (1200 -> 200 Hz)
        freq = 1200 - 2000 * t
        tone = math.sin(2 * math.pi * freq * t) * 0.4
        tone += math.sin(2 * math.pi * freq * 2.5 * t) * 0.2
        val = prev * 0.5 + tone * 0.5
        samples.append(_clamp(val * env * MAX_16BIT * 0.45))
    return _make_sound(samples)


def _gen_level_up() -> pygame.mixer.Sound:
    """Level up: fanfara rapida ascendente."""
    duration = 0.5
    n = int(SAMPLE_RATE * duration)
    samples = []
    # Arpeggio maggiore veloce
    notes = [523, 659, 784, 1047, 1319]  # Do5-Mi5-Sol5-Do6-Mi6
    note_dur = duration / len(notes)
    for i in range(n):
        t = i / SAMPLE_RATE
        note_idx = min(int(t / note_dur), len(notes) - 1)
        freq = notes[note_idx]
        local_t = t - note_idx * note_dur
        env = max(0.0, 1.0 - local_t / note_dur * 0.5)
        # Tono brillante
        val = math.sin(2 * math.pi * freq * t) * 0.5
        val += math.sin(2 * math.pi * freq * 2 * t) * 0.3
        val += math.sin(2 * math.pi * freq * 3 * t) * 0.1
        g_env = min(1.0, t / 0.005)
        samples.append(_clamp(val * env * g_env * MAX_16BIT * 0.25))
    return _make_sound(samples)


# ═══════════════════════════════════════════════════════════
#  Generazione musica di sottofondo
# ═══════════════════════════════════════════════════════════

def _gen_bgm() -> pygame.mixer.Sound:
    """Genera un loop musicale procedurale (~16 secondi).

    Stile: retro sci-fi ambient con basso pulsante e arpeggi.
    """
    bpm = 110
    beat_dur = 60.0 / bpm
    bars = 8
    beats_per_bar = 4
    total_beats = bars * beats_per_bar
    duration = total_beats * beat_dur
    n = int(SAMPLE_RATE * duration)
    samples = []

    # Progressione accordi (semitoni da Do3=131 Hz)
    # Am - F - C - G (ripetuto 2x)
    chord_prog = [
        (220.0, 261.6, 329.6),   # Am: La3, Do4, Mi4
        (174.6, 220.0, 261.6),   # F:  Fa3, La3, Do4
        (261.6, 329.6, 392.0),   # C:  Do4, Mi4, Sol4
        (196.0, 246.9, 293.7),   # G:  Sol3, Si3, Re4
    ] * 2  # ripetuto per 8 battute

    # Linea di basso (nota fondamentale dell'accordo, un'ottava sotto)
    bass_notes = [110.0, 87.3, 130.8, 98.0] * 2

    for i in range(n):
        t = i / SAMPLE_RATE
        beat = t / beat_dur
        bar = int(beat / beats_per_bar) % bars
        beat_in_bar = beat % beats_per_bar

        chord = chord_prog[bar]
        bass_freq = bass_notes[bar]

        val = 0.0

        # ── Basso pulsante (onda quadra morbida) ──
        bass_phase = (bass_freq * t) % 1.0
        bass_wave = 1.0 if bass_phase < 0.5 else -1.0
        # Addolcisci la quadra
        bass_wave = bass_wave * 0.6 + math.sin(2 * math.pi * bass_freq * t) * 0.4
        # Pulse: volume più alto su beat 1 e 3
        beat_pos = beat_in_bar % 1.0
        bass_env = 0.7 if int(beat_in_bar) % 2 == 0 else 0.4
        # Envelope pulsante dentro ogni beat
        bass_env *= max(0.0, 1.0 - beat_pos * 0.8)
        val += bass_wave * bass_env * 0.2

        # ── Pad accordi (sinusoidi sovrapposte, lento) ──
        pad = 0.0
        for note_freq in chord:
            pad += math.sin(2 * math.pi * note_freq * t)
            # Leggero detune per larghezza
            pad += math.sin(2 * math.pi * note_freq * 1.003 * t) * 0.5
        pad /= len(chord) * 1.5
        # Volume pad costante e basso
        val += pad * 0.08

        # ── Arpeggio (note dell'accordo in sequenza su ogni 16esimo) ──
        sixteenth = int(beat_in_bar * 4) % 4
        arp_freq = chord[sixteenth % len(chord)]
        # Ottava su per brillantezza
        arp_freq *= 2
        arp_wave = math.sin(2 * math.pi * arp_freq * t) * 0.5
        arp_wave += math.sin(2 * math.pi * arp_freq * 2 * t) * 0.2
        # Envelope per ogni 16esimo
        sixteenth_pos = (beat_in_bar * 4) % 1.0
        arp_env = max(0.0, 1.0 - sixteenth_pos * 1.5)
        # Alterno volume su pattern ritmico (accento sul 1° e 3° sedicesimo)
        accent = 1.0 if sixteenth in (0, 2) else 0.6
        val += arp_wave * arp_env * accent * 0.1

        # ── Hi-hat sintetico (rumore filtrato su 8vi) ──
        eighth = int(beat_in_bar * 2) % 2
        eighth_pos = (beat_in_bar * 2) % 1.0
        if eighth_pos < 0.15:
            hh_env = max(0.0, 1.0 - eighth_pos / 0.15)
            # Rumore ad alta frequenza
            hh = math.sin(2 * math.pi * 8000 * t + i * 0.7) * 0.3
            hh += math.sin(2 * math.pi * 11000 * t + i * 1.3) * 0.3
            hh += math.sin(2 * math.pi * 6500 * t) * 0.2
            val += hh * hh_env * 0.06

        # ── Kick sintetico su beat 1 e 3 ──
        if int(beat_in_bar) % 2 == 0:
            kick_pos = beat_in_bar % 1.0
            if kick_pos < 0.12:
                kick_env = max(0.0, 1.0 - kick_pos / 0.12) ** 2
                kick_freq = 150 - 800 * kick_pos
                kick = math.sin(2 * math.pi * kick_freq * t) * kick_env
                val += kick * 0.15

        samples.append(_clamp(val * MAX_16BIT))

    return _make_sound(samples)


# ═══════════════════════════════════════════════════════════
#  Sound Manager
# ═══════════════════════════════════════════════════════════

class SoundManager:
    """Gestisce tutti i suoni del gioco. Genera tutto proceduralmente."""

    def __init__(self):
        pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=1, buffer=512)
        pygame.mixer.set_num_channels(16)

        # Genera gli effetti
        self.sfx_shoot = _gen_shoot()
        self.sfx_explosion = _gen_explosion()
        self.sfx_powerup = _gen_powerup()
        self.sfx_player_hit = _gen_player_hit()
        self.sfx_game_over = _gen_game_over()
        self.sfx_missile = _gen_missile_launch()
        self.sfx_bomb = _gen_bomb()
        self.sfx_level_up = _gen_level_up()
        self.sfx_shield_hit = _gen_shield_hit()
        self.sfx_shield_break = _gen_shield_break()

        # Genera la musica di sottofondo
        self.bgm = _gen_bgm()

        # Volumi
        self.sfx_shoot.set_volume(0.4)
        self.sfx_explosion.set_volume(0.5)
        self.sfx_powerup.set_volume(0.6)
        self.sfx_player_hit.set_volume(0.6)
        self.sfx_game_over.set_volume(0.7)
        self.sfx_missile.set_volume(0.5)
        self.sfx_bomb.set_volume(0.6)
        self.sfx_level_up.set_volume(0.6)
        self.sfx_shield_hit.set_volume(0.5)
        self.sfx_shield_break.set_volume(0.7)
        self.bgm.set_volume(0.25)

        # Canale dedicato per la musica (loop)
        self._bgm_channel = pygame.mixer.Channel(0)

        # Flag
        self._music_playing = False

    def play_shoot(self):
        self.sfx_shoot.play()

    def play_explosion(self):
        self.sfx_explosion.play()

    def play_powerup(self):
        self.sfx_powerup.play()

    def play_player_hit(self):
        self.sfx_player_hit.play()

    def play_game_over(self):
        self.sfx_game_over.play()

    def play_missile(self):
        self.sfx_missile.play()

    def play_bomb(self):
        self.sfx_bomb.play()

    def play_level_up(self):
        self.sfx_level_up.play()

    def play_shield_hit(self):
        self.sfx_shield_hit.play()

    def play_shield_break(self):
        self.sfx_shield_break.play()

    def start_music(self):
        """Avvia la musica di sottofondo in loop."""
        if not self._music_playing:
            self._bgm_channel.play(self.bgm, loops=-1)
            self._music_playing = True

    def stop_music(self):
        """Ferma la musica."""
        self._bgm_channel.stop()
        self._music_playing = False

    def pause_music(self):
        self._bgm_channel.pause()

    def resume_music(self):
        self._bgm_channel.unpause()
