import random
import pygame
from settings import (
    SCREEN_WIDTH, SCREEN_HEIGHT,
    MISSILE_SPEED, MISSILE_KILL_COUNT,
    MISSILE_POWERUP_SPEED,
)


class MissilePowerUp(pygame.sprite.Sprite):
    """Power-up raccoglibile che dà un missile al giocatore."""

    def __init__(self):
        super().__init__()
        self.image = self._create_sprite()
        self.rect = self.image.get_rect(
            center=(
                SCREEN_WIDTH + 30,
                random.randint(60, SCREEN_HEIGHT - 60),
            )
        )
        self.speed = MISSILE_POWERUP_SPEED
        self._float_tick = 0

    def _create_sprite(self) -> pygame.Surface:
        """Icona missile pixel-art con sfondo luminoso."""
        PX = 2
        gw, gh = 16, 16
        surf = pygame.Surface((gw * PX, gh * PX), pygame.SRCALPHA)

        def px(x, y, color):
            if 0 <= x < gw and 0 <= y < gh:
                pygame.draw.rect(surf, color, (x * PX, y * PX, PX, PX))

        def hline(x1, x2, y, color):
            for x in range(x1, x2 + 1):
                px(x, y, color)

        def fill(x1, y1, x2, y2, color):
            for y in range(y1, y2 + 1):
                hline(x1, x2, y, color)

        # Alone luminoso giallo/arancione
        GLOW = (255, 200, 50, 60)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if abs(dx) + abs(dy) <= 3:
                    cx, cy = 8 + dx, 8 + dy
                    if 0 <= cx < gw and 0 <= cy < gh:
                        pygame.draw.rect(surf, GLOW, (cx * PX, cy * PX, PX, PX))

        # Corpo missile (orizzontale, punta a destra)
        BODY = (200, 60, 40)
        BODY_L = (240, 90, 50)
        NOSE = (220, 220, 220)
        FIN = (160, 45, 30)
        FLAME = (255, 180, 50)

        # Fusoliera
        fill(4, 7, 11, 9, BODY)
        hline(4, 11, 7, BODY_L)
        # Ogiva
        fill(12, 7, 13, 9, NOSE)
        px(14, 8, NOSE)
        # Alette posteriori
        px(4, 5, FIN)
        px(4, 6, FIN)
        px(4, 10, FIN)
        px(4, 11, FIN)
        px(3, 5, FIN)
        px(3, 11, FIN)
        # Fiamma
        px(3, 8, FLAME)
        px(2, 8, (255, 220, 100))
        px(2, 7, FLAME)
        px(2, 9, FLAME)

        return surf

    def update(self, dt: float):
        self.rect.x -= int(self.speed * dt)
        # Leggero fluttuamento verticale
        self._float_tick += 0.1
        import math
        self.rect.y += int(math.sin(self._float_tick) * 0.5)
        if self.rect.right < 0:
            self.kill()


class Missile(pygame.sprite.Sprite):
    """Missile lanciato dal giocatore: vola verso destra e cerca i nemici."""

    def __init__(self, x: int, y: int, enemies: pygame.sprite.Group):
        super().__init__()
        self.image = self._create_sprite()
        self.rect = self.image.get_rect(center=(x, y))
        self.speed = MISSILE_SPEED
        self.kills_left = MISSILE_KILL_COUNT
        self.enemies = enemies
        self.total_kills = 0
        self.kill_positions: list[tuple[int, int]] = []

    def _create_sprite(self) -> pygame.Surface:
        """Missile in volo - più grande del proiettile normale."""
        PX = 2
        gw, gh = 14, 5
        surf = pygame.Surface((gw * PX, gh * PX), pygame.SRCALPHA)

        def px(x, y, color):
            if 0 <= x < gw and 0 <= y < gh:
                pygame.draw.rect(surf, color, (x * PX, y * PX, PX, PX))

        def hline(x1, x2, y, color):
            for x in range(x1, x2 + 1):
                px(x, y, color)

        BODY = (200, 60, 40)
        BODY_L = (240, 90, 50)
        NOSE = (220, 220, 220)
        FLAME = (255, 180, 50)

        # Corpo
        hline(3, 10, 1, BODY_L)
        hline(3, 10, 2, BODY)
        hline(3, 10, 3, BODY)
        # Ogiva
        hline(11, 12, 2, NOSE)
        px(13, 2, NOSE)
        # Alette
        px(3, 0, BODY)
        px(3, 4, BODY)
        # Fiamma
        px(2, 2, FLAME)
        px(1, 2, (255, 220, 100))
        px(0, 2, (255, 140, 40))
        px(1, 1, (255, 140, 40))
        px(1, 3, (255, 140, 40))

        return surf

    def update(self, dt: float):
        self.rect.x += int(self.speed * dt)

        # Controlla collisione con nemici
        if self.kills_left > 0:
            hit_list = pygame.sprite.spritecollide(self, self.enemies, False)
            for enemy in hit_list:
                if self.kills_left <= 0:
                    break
                self.kill_positions.append((enemy.rect.centerx, enemy.rect.centery))
                enemy.kill()
                self.kills_left -= 1
                self.total_kills += 1

        if self.kills_left <= 0 or self.rect.left > SCREEN_WIDTH:
            self.kill()
