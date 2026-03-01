import pygame
from settings import SCREEN_WIDTH, BULLET_SPEED, BULLET_SIZE


class Bullet(pygame.sprite.Sprite):
    """Proiettile infuocato con scia rosso/arancione."""

    def __init__(self, x: int, y: int):
        super().__init__()
        self._create_frames()
        self.frame = 0
        self.image = self.frames[0]
        self.rect = self.image.get_rect(center=(x, y))
        self.speed = BULLET_SPEED

    def _create_frames(self):
        """Crea 3 frame di animazione per la scia tremolante."""
        PX = 2
        gw, gh = 18, 5  # griglia logica
        self.frames = []

        # Variazioni della scia per ogni frame
        trail_variants = [
            # Frame 0
            [(-8, 0), (-7, -1), (-7, 1), (-6, 0), (-5, -1), (-5, 1)],
            # Frame 1
            [(-8, 1), (-7, 0), (-7, -1), (-6, 1), (-5, 0), (-6, -1)],
            # Frame 2
            [(-8, -1), (-7, 0), (-7, 1), (-6, -1), (-5, 0), (-5, 1)],
        ]

        for variant_idx in range(3):
            surf = pygame.Surface((gw * PX, gh * PX), pygame.SRCALPHA)

            def px(x, y, color):
                sx, sy = x + 8, y + 2  # offset per centrare (scia va a sinistra)
                if 0 <= sx < gw and 0 <= sy < gh:
                    pygame.draw.rect(surf, color, (sx * PX, sy * PX, PX, PX))

            # ── Punta del proiettile (ogiva metallica) ──
            px(9, 0, (220, 220, 210))
            px(8, 0, (180, 175, 160))
            px(8, -1, (160, 155, 140))
            px(8, 1, (160, 155, 140))

            # ── Corpo proiettile (dorato/ottone) ──
            px(7, 0, (200, 160, 60))
            px(6, 0, (190, 145, 50))
            px(5, 0, (180, 135, 45))
            px(7, -1, (170, 130, 45))
            px(7, 1, (170, 130, 45))
            px(6, -1, (160, 120, 40))
            px(6, 1, (160, 120, 40))

            # ── Fiamma interna (giallo brillante) ──
            px(4, 0, (255, 240, 120))
            px(3, 0, (255, 220, 80))
            px(4, -1, (255, 200, 60))
            px(4, 1, (255, 200, 60))

            # ── Fiamma media (arancione) ──
            px(2, 0, (255, 160, 30))
            px(1, 0, (255, 130, 20))
            px(3, -1, (255, 150, 30))
            px(3, 1, (255, 150, 30))
            px(2, -1, (250, 120, 20))
            px(2, 1, (250, 120, 20))

            # ── Scia rossa (tremolante, varia per frame) ──
            TRAIL_RED = (220, 50, 20, 200)
            TRAIL_DK = (180, 30, 10, 150)
            TRAIL_FADE = (150, 25, 10, 100)

            px(0, 0, (240, 80, 15))
            px(-1, 0, TRAIL_RED)
            px(-2, 0, TRAIL_DK)
            px(-3, 0, TRAIL_FADE)
            px(1, -1, (230, 70, 15))
            px(1, 1, (230, 70, 15))

            # Particelle scia variabili
            for dx, dy in trail_variants[variant_idx]:
                px(dx, dy, TRAIL_DK)

            # Particelle più lontane sfumate
            px(-4, 0, (120, 20, 10, 70))
            if variant_idx == 0:
                px(-5, -1, (100, 15, 8, 50))
            elif variant_idx == 1:
                px(-5, 1, (100, 15, 8, 50))
            else:
                px(-5, 0, (100, 15, 8, 50))

            self.frames.append(surf)

    def update(self, dt: float):
        self.rect.x += int(self.speed * dt)
        # Anima la scia (cicla tra i 3 frame)
        self.frame = (self.frame + 1) % 3
        self.image = self.frames[self.frame]
        if self.rect.left > SCREEN_WIDTH:
            self.kill()
