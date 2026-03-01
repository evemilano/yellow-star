"""Power-up droppato dai nemici — 4 categorie di potenziamento."""

import math
import random
import pygame
from settings import SCREEN_HEIGHT, POWERUP_SPEED

# Colori per categoria
_COLORS = {
    "engine": (80, 220, 80),      # verde
    "weapon": (255, 150, 50),     # arancione
    "shield": (80, 150, 255),     # blu
    "bomb":   (200, 80, 255),     # viola
}

# Lettere icona
_LABELS = {
    "engine": "M",   # Motore
    "weapon": "W",   # Weapon
    "shield": "S",   # Scudo
    "bomb":   "B",   # Bomba
}

# Cache sprite (una volta per tipo)
_sprite_cache: dict[str, pygame.Surface] = {}


def _create_powerup_sprite(ptype: str) -> pygame.Surface:
    """Crea un'icona power-up con alone colorato e lettera."""
    if ptype in _sprite_cache:
        return _sprite_cache[ptype]

    size = 32
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    center = size // 2
    color = _COLORS[ptype]
    glow = (*color, 50)

    # Alone esterno
    pygame.draw.circle(surf, glow, (center, center), 15)
    # Cerchio pieno
    pygame.draw.circle(surf, color, (center, center), 10)
    # Bordo bianco
    pygame.draw.circle(surf, (255, 255, 255), (center, center), 10, 2)
    # Lettera
    font = pygame.font.Font(None, 20)
    label = font.render(_LABELS[ptype], True, (255, 255, 255))
    label_rect = label.get_rect(center=(center, center))
    surf.blit(label, label_rect)

    _sprite_cache[ptype] = surf
    return surf


class PowerUp(pygame.sprite.Sprite):
    """Power-up droppato da un nemico distrutto."""

    def __init__(self, x: int, y: int, ptype: str | None = None):
        super().__init__()
        self.ptype = ptype or random.choice(list(_COLORS.keys()))
        self.image = _create_powerup_sprite(self.ptype)
        self.rect = self.image.get_rect(center=(x, y))
        self.speed = POWERUP_SPEED
        self._float_tick = random.uniform(0, math.pi * 2)

    def update(self, dt: float):
        self.rect.x -= int(self.speed * dt)
        # Fluttuamento verticale
        self._float_tick += 0.1
        self.rect.y += int(math.sin(self._float_tick) * 0.8)
        # Clampa verticalmente
        self.rect.clamp_ip(pygame.Rect(0, 0, 9999, SCREEN_HEIGHT))
        if self.rect.right < 0:
            self.kill()
