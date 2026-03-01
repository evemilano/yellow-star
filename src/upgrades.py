"""Gestione dei livelli di potenziamento (0-10) per le 4 categorie."""

from settings import PLAYER_SPEED, PLAYER_SHOOT_DELAY, PLAYER_INVINCIBLE_TIME

MAX_UPGRADE_LEVEL = 10


class UpgradeManager:
    """Tiene traccia dei livelli di potenziamento e calcola i parametri."""

    def __init__(self):
        self.levels = {
            "engine": 0,
            "weapon": 0,
            "shield": 0,
            "bomb": 0,
        }

    def upgrade(self, category: str):
        """Incrementa il livello di una categoria (max 10)."""
        if category in self.levels and self.levels[category] < MAX_UPGRADE_LEVEL:
            self.levels[category] += 1

    def reset(self):
        """Reset a livello 0 per tutte le categorie."""
        for key in self.levels:
            self.levels[key] = 0

    # ── Engine ──

    @property
    def speed_multiplier(self) -> float:
        """Moltiplicatore velocità: da 1.0 (lv0) a 2.0 (lv10)."""
        return 1.0 + self.levels["engine"] * 0.1

    @property
    def player_speed(self) -> float:
        return PLAYER_SPEED * self.speed_multiplier

    # ── Weapon ──

    @property
    def bullet_count(self) -> int:
        """Numero di proiettili sparati contemporaneamente."""
        lv = self.levels["weapon"]
        return min(5, 1 + lv // 2)

    @property
    def shoot_delay(self) -> int:
        """Delay tra un colpo e l'altro in ms."""
        lv = self.levels["weapon"]
        base = PLAYER_SHOOT_DELAY
        if lv >= 10:
            return int(base * 0.70)  # -30%
        elif lv % 2 == 1:  # livelli dispari
            return int(base * 0.85)  # -15%
        return base

    def get_bullet_offsets(self) -> list[int]:
        """Lista di offset Y per i proiettili (rispetto al centro del player).

        Livelli pari aggiungono proiettili, dispari aumentano cadenza.
        """
        n = self.bullet_count
        if n == 1:
            return [0]
        elif n == 2:
            return [-8, 8]
        elif n == 3:
            return [-12, 0, 12]
        elif n == 4:
            return [-16, -6, 6, 16]
        else:  # n == 5
            return [-18, -9, 0, 9, 18]

    # ── Shield ──

    @property
    def shield_max(self) -> int:
        """Punti scudo massimi al livello attuale."""
        lv = self.levels["shield"]
        if lv == 0:
            return 0
        return (lv + 1) // 2  # 1,1,2,2,3,3,4,4,5,5

    @property
    def invincible_time(self) -> int:
        """Durata invincibilità post-danno in ms."""
        lv = self.levels["shield"]
        if lv < 2:
            return PLAYER_INVINCIBLE_TIME  # 2000 default
        return PLAYER_INVINCIBLE_TIME + ((lv // 2) * 500)

    # ── Bomb ──

    @property
    def max_bombs(self) -> int:
        """Numero massimo di bombe trasportabili."""
        lv = self.levels["bomb"]
        if lv == 0:
            return 0
        return (lv + 1) // 2  # 1,1,2,2,3,3,4,4,5,5

    @property
    def bomb_destroys_asteroids(self) -> bool:
        return self.levels["bomb"] >= 2

    @property
    def bomb_double_score(self) -> bool:
        return self.levels["bomb"] >= 6

    @property
    def bomb_freeze_spawn(self) -> bool:
        return self.levels["bomb"] >= 8

    @property
    def bomb_shield_field(self) -> bool:
        return self.levels["bomb"] >= 10
