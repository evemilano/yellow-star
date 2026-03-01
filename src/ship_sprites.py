"""Sprite dell'astronave del giocatore per i 10 livelli weapon.

Carica i PNG da assets/images/player/ship_01.png .. ship_10.png.
Livello 0 (nessun power-up) usa lo sprite di livello 1.
Gli sprite vengono scalati per adattarsi al gioco (larghezza target crescente).
"""

import os
import pygame

_SPRITES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "assets", "images", "player",
)

_cache: dict[int, pygame.Surface] = {}

# Larghezza target per ogni livello (crescente da ~90 a ~200)
_TARGET_WIDTH = {
    1: 58,
    2: 65,
    3: 72,
    4: 75,
    5: 81,
    6: 91,
    7: 101,
    8: 114,
    9: 120,
    10: 130,
}


def create_ship_sprite(weapon_level: int) -> pygame.Surface:
    """Restituisce lo sprite dell'astronave per il livello weapon dato (0-10)."""
    level = max(1, min(weapon_level, 10))

    if level in _cache:
        return _cache[level]

    path = os.path.join(_SPRITES_DIR, f"ship_{level:02d}.png")
    raw = pygame.image.load(path)
    try:
        raw = raw.convert_alpha()
    except pygame.error:
        pass

    # Scala alla larghezza target mantenendo le proporzioni
    target_w = _TARGET_WIDTH[level]
    orig_w, orig_h = raw.get_size()
    scale = target_w / orig_w
    new_h = int(orig_h * scale)
    surf = pygame.transform.scale(raw, (target_w, new_h))

    _cache[level] = surf
    return surf
