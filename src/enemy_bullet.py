"""Proiettile sparato dalle astronavi nemiche."""

import math
import pygame
from settings import SCREEN_WIDTH, SCREEN_HEIGHT, ENEMY_BULLET_SPEED


class EnemyBullet(pygame.sprite.Sprite):
    """Proiettile nemico magenta/viola con scia animata (3 frame)."""

    def __init__(self, x: int, y: int, angle_deg: float = 180.0):
        super().__init__()
        self._create_frames()
        self.frame = 0
        self.image = self.frames[0]
        self.rect = self.image.get_rect(center=(x, y))

        rad = math.radians(angle_deg)
        self.vx = math.cos(rad) * ENEMY_BULLET_SPEED
        self.vy = math.sin(rad) * ENEMY_BULLET_SPEED

    def _create_frames(self):
        """3 frame animati: corpo magenta con scia viola tremolante."""
        PX = 2
        gw, gh = 16, 5
        self.frames = []

        trail_variants = [
            [(8, 0), (7, -1), (7, 1), (6, 0)],
            [(8, 1), (7, 0), (7, -1), (6, 1)],
            [(8, -1), (7, 0), (7, 1), (6, -1)],
        ]

        for variant_idx in range(3):
            surf = pygame.Surface((gw * PX, gh * PX), pygame.SRCALPHA)

            def px(x, y, color):
                sx, sy = x, y + 2
                if 0 <= sx < gw and 0 <= sy < gh:
                    pygame.draw.rect(surf, color, (sx * PX, sy * PX, PX, PX))

            # Punta (lato sinistro — direzione di volo)
            px(0, 0, (255, 200, 220))
            px(1, 0, (230, 100, 180))
            px(1, -1, (200, 80, 160))
            px(1, 1, (200, 80, 160))

            # Corpo magenta
            px(2, 0, (220, 60, 150))
            px(3, 0, (200, 50, 130))
            px(4, 0, (180, 40, 120))
            px(2, -1, (180, 45, 130))
            px(2, 1, (180, 45, 130))
            px(3, -1, (160, 35, 110))
            px(3, 1, (160, 35, 110))

            # Nucleo luminoso
            px(5, 0, (240, 120, 200, 220))
            px(6, 0, (200, 80, 180, 200))

            # Scia viola
            TRAIL2 = (120, 20, 110, 150)
            TRAIL3 = (80, 10, 80, 90)

            px(7, 0, (170, 40, 150))
            for dx, dy in trail_variants[variant_idx]:
                px(dx, dy, TRAIL2)
            px(9, 0, TRAIL2)
            px(10, 0, TRAIL3)
            px(11, 0, (60, 5, 60, 50))

            self.frames.append(surf)

    def update(self, dt: float):
        self.rect.x += int(self.vx * dt)
        self.rect.y += int(self.vy * dt)

        self.frame = (self.frame + 1) % 3
        self.image = self.frames[self.frame]

        if (self.rect.right < 0 or self.rect.left > SCREEN_WIDTH
                or self.rect.bottom < 0 or self.rect.top > SCREEN_HEIGHT):
            self.kill()
