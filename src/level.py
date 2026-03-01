"""Sistema di livelli a difficoltà crescente."""

import random
from settings import (
    ENEMY_SPAWN_DELAY, ENEMY_SPEED_MIN, ENEMY_SPEED_MAX,
    ASTEROID_SPAWN_DELAY, ASTEROID_SPEED_MIN, ASTEROID_SPEED_MAX,
    ASTEROID_SIZE_MIN, ASTEROID_SIZE_MAX,
    MISSILE_POWERUP_SPAWN_DELAY,
    LEVEL_SCORE_BASE, LEVEL_SCORE_GROWTH,
    LEVEL_DIFFICULTY_FACTOR,
    LEVEL_MAX,
    ENEMY_SHOOTER_CHANCE, ENEMY_SHOOTER_UNLOCK,
    ENEMY_SHOOT_DELAY_BASIC, ENEMY_SHOOT_DELAY_AIMED,
    ENEMY_SHOOT_DELAY_BURST, ENEMY_SHOOT_DELAY_FAST,
)

_BASE_SHOOT_DELAYS = {
    "basic": ENEMY_SHOOT_DELAY_BASIC,
    "aimed": ENEMY_SHOOT_DELAY_AIMED,
    "burst": ENEMY_SHOOT_DELAY_BURST,
    "fast":  ENEMY_SHOOT_DELAY_FAST,
}


class LevelManager:
    """Gestisce la progressione dei livelli e i parametri di difficoltà.

    La difficoltà scala con una formula esponenziale morbida:
    ogni livello moltiplica spawn delay × (1 / factor) e velocità × factor.
    """

    def __init__(self):
        self.level = 1
        self.levelup_timer = 0.0  # secondi rimasti per mostrare "LEVEL UP"
        self._cache_params()

    def _score_for_level(self, level: int) -> int:
        """Score necessario per raggiungere il livello dato."""
        if level <= 1:
            return 0
        # Progressione: base, base+base*growth, base+base*growth+base*growth^2 ...
        total = 0
        for i in range(1, level):
            total += int(LEVEL_SCORE_BASE * (LEVEL_SCORE_GROWTH ** (i - 1)))
        return total

    def _cache_params(self):
        """Calcola e salva i parametri di difficoltà per il livello corrente."""
        n = self.level - 1  # 0-based offset
        f = LEVEL_DIFFICULTY_FACTOR ** n

        # Spawn delay: diminuisce (più nemici/asteroidi), con un floor
        self.enemy_spawn_delay = max(400, int(ENEMY_SPAWN_DELAY / f))
        self.asteroid_spawn_delay = max(800, int(ASTEROID_SPAWN_DELAY / f))
        self.powerup_spawn_delay = min(15000, int(MISSILE_POWERUP_SPAWN_DELAY * f))

        # Velocità nemici: aumenta
        self.enemy_speed_min = ENEMY_SPEED_MIN + n * 0.4
        self.enemy_speed_max = ENEMY_SPEED_MAX + n * 0.5

        # Velocità asteroidi: aumenta
        self.asteroid_speed_min = ASTEROID_SPEED_MIN + n * 0.3
        self.asteroid_speed_max = ASTEROID_SPEED_MAX + n * 0.4

        # Dimensione asteroidi: leggermente più grandi
        self.asteroid_size_min = min(40, ASTEROID_SIZE_MIN + n * 2)
        self.asteroid_size_max = min(65, ASTEROID_SIZE_MAX + n * 2)

        # Score necessario per il prossimo livello
        if self.level < LEVEL_MAX:
            self.next_level_score = self._score_for_level(self.level + 1)
        else:
            self.next_level_score = float('inf')

        # ── Parametri sparo nemici ──
        idx = min(self.level - 1, len(ENEMY_SHOOTER_CHANCE) - 1)
        self.enemy_shooter_chance = ENEMY_SHOOTER_CHANCE[idx]

        self.available_shooter_types = [
            stype for stype, unlock_lv in ENEMY_SHOOTER_UNLOCK.items()
            if self.level >= unlock_lv
        ]

        self.shoot_delays = {
            stype: max(600, int(base / f))
            for stype, base in _BASE_SHOOT_DELAYS.items()
        }

    def pick_shoot_config(self) -> tuple:
        """Ritorna (shoot_type, delay_ms) per un nemico appena spawnato.
        Ritorna ("none", 0) se il nemico non deve sparare."""
        if not self.available_shooter_types:
            return ("none", 0)
        if random.random() >= self.enemy_shooter_chance:
            return ("none", 0)
        stype = random.choice(self.available_shooter_types)
        return (stype, self.shoot_delays[stype])

    def check_levelup(self, score: int) -> bool:
        """Controlla se lo score ha raggiunto la soglia per il prossimo livello.
        Ritorna True se si è saliti di livello."""
        if self.level >= LEVEL_MAX:
            return False
        if score >= self.next_level_score:
            self.level += 1
            self.levelup_timer = 2.5  # secondi di visualizzazione
            self._cache_params()
            return True
        return False

    def update(self, dt: float, fps: float):
        """Aggiorna il timer del messaggio LEVEL UP."""
        if self.levelup_timer > 0:
            self.levelup_timer -= dt / fps

    def reset(self):
        """Reset al livello 1 (per restart)."""
        self.level = 1
        self.levelup_timer = 0.0
        self._cache_params()
