import os
import math
import random
import pygame
from settings import (
    SCREEN_WIDTH, SCREEN_HEIGHT,
    ENEMY_SPEED_MIN, ENEMY_SPEED_MAX, ENEMY_MODELS,
    ENEMY_BURST_ANGLES,
)
from src.enemy_bullet import EnemyBullet

# Cache degli sprite caricati (condivisa tra tutte le istanze)
_enemy_sprites: list[pygame.Surface] = []


def _load_enemy_sprites():
    """Carica tutti i 16 modelli di navi nemiche da file PNG."""
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "assets", "images", "enemies")
    for i in range(1, ENEMY_MODELS + 1):
        path = os.path.join(base, f"enemy_{i:02d}.png")
        img = pygame.image.load(path).convert_alpha()
        _enemy_sprites.append(img)


class Enemy(pygame.sprite.Sprite):
    def __init__(self, speed_min=None, speed_max=None,
                 shoot_type="none", shoot_delay=2000,
                 enemy_bullets=None, player=None):
        super().__init__()
        # Carica gli sprite la prima volta
        if not _enemy_sprites:
            _load_enemy_sprites()

        self.image = random.choice(_enemy_sprites)
        w, h = self.image.get_size()
        self.rect = self.image.get_rect(
            center=(
                SCREEN_WIDTH + w,
                random.randint(h, SCREEN_HEIGHT - h),
            )
        )
        smin = speed_min if speed_min is not None else ENEMY_SPEED_MIN
        smax = speed_max if speed_max is not None else ENEMY_SPEED_MAX
        self.speed = random.uniform(smin, smax)

        # Stato sparo
        self.shoot_type = shoot_type
        self.shoot_delay = shoot_delay
        self.enemy_bullets = enemy_bullets
        self.player = player
        # Stagger iniziale per evitare muri di proiettili simultanei
        self.last_shot = pygame.time.get_ticks() + random.randint(500, 1500)

    def _fire(self):
        """Genera proiettili in base al tipo di shooter."""
        if self.enemy_bullets is None:
            return

        cx, cy = self.rect.left, self.rect.centery

        if self.shoot_type == "basic" or self.shoot_type == "fast":
            self.enemy_bullets.add(EnemyBullet(cx, cy, 180))

        elif self.shoot_type == "aimed":
            if self.player is not None:
                dx = self.player.rect.centerx - cx
                dy = self.player.rect.centery - cy
                angle = math.degrees(math.atan2(dy, dx))
            else:
                angle = 180
            self.enemy_bullets.add(EnemyBullet(cx, cy, angle))

        elif self.shoot_type == "burst":
            for a_offset in ENEMY_BURST_ANGLES:
                self.enemy_bullets.add(EnemyBullet(cx, cy, 180 + a_offset))

    def update(self, dt: float):
        self.rect.x -= int(self.speed * dt)
        if self.rect.right < 0:
            self.kill()
            return

        # Spara solo se il nemico è visibile sullo schermo
        if self.shoot_type != "none" and self.rect.right <= SCREEN_WIDTH:
            now = pygame.time.get_ticks()
            if now - self.last_shot >= self.shoot_delay:
                self.last_shot = now
                self._fire()
