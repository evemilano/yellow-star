import random
import pygame
from settings import (
    SCREEN_WIDTH, SCREEN_HEIGHT,
    SKY_BLUE, LIGHT_BLUE,
    MOUNTAIN_FAR, MOUNTAIN_MID, MOUNTAIN_NEAR,
    GREEN, DARK_GREEN, BROWN, GROUND_COLOR,
    PARALLAX_MOUNTAIN_FAR_SPEED, PARALLAX_MOUNTAIN_MID_SPEED,
    PARALLAX_MOUNTAIN_NEAR_SPEED, PARALLAX_TREES_SPEED, PARALLAX_GROUND_SPEED,
)


class ScrollingLayer:
    """Un layer orizzontale che scorre e si ripete (tile seamless)."""

    def __init__(self, surface: pygame.Surface, speed: float, y_offset: int = 0):
        self.surface = surface
        self.speed = speed
        self.width = surface.get_width()
        self.y_offset = y_offset
        self.x = 0.0

    def update(self, dt: float):
        self.x -= self.speed * dt
        if self.x <= -self.width:
            self.x += self.width

    def draw(self, screen: pygame.Surface):
        ix = int(self.x)
        screen.blit(self.surface, (ix, self.y_offset))
        screen.blit(self.surface, (ix + self.width, self.y_offset))


def _generate_mountains(width: int, height: int, color: tuple,
                        peak_min: int, peak_max: int, segments: int) -> pygame.Surface:
    """Genera una striscia di montagne procedurali."""
    surf = pygame.Surface((width, height), pygame.SRCALPHA)
    step = width // segments
    points = [(0, height)]
    for i in range(segments + 1):
        x = i * step
        y = random.randint(peak_min, peak_max)
        points.append((x, y))
    points.append((width, height))
    pygame.draw.polygon(surf, color, points)
    return surf


def _generate_trees(width: int, total_height: int, ground_y: int,
                    color_trunk: tuple, color_leaves: tuple,
                    count: int) -> pygame.Surface:
    """Genera alberi procedurali con tronco e chioma tonda."""
    surf = pygame.Surface((width, total_height), pygame.SRCALPHA)
    for _ in range(count):
        x = random.randint(0, width)
        trunk_h = random.randint(25, 50)
        trunk_w = random.randint(4, 8)
        canopy_r = random.randint(14, 28)
        trunk_top = ground_y - trunk_h
        # Tronco
        pygame.draw.rect(surf, color_trunk,
                         (x - trunk_w // 2, trunk_top, trunk_w, trunk_h))
        # Chioma (2-3 cerchi sovrapposti)
        for _ in range(random.randint(2, 3)):
            ox = random.randint(-canopy_r // 3, canopy_r // 3)
            oy = random.randint(-canopy_r // 3, canopy_r // 4)
            pygame.draw.circle(surf, color_leaves,
                               (x + ox, trunk_top + oy), canopy_r)
    return surf


def _generate_ground(width: int, height: int, color: tuple) -> pygame.Surface:
    """Genera il terreno con un profilo leggermente ondulato."""
    surf = pygame.Surface((width, height), pygame.SRCALPHA)
    points = [(0, height)]
    segments = 30
    step = width // segments
    for i in range(segments + 1):
        x = i * step
        y = random.randint(0, 12)
        points.append((x, y))
    points.append((width, height))
    pygame.draw.polygon(surf, color, points)
    return surf


class ParallaxBackground:
    """Sfondo parallasse con cielo, montagne, alberi e terreno."""

    def __init__(self):
        tile_w = SCREEN_WIDTH * 2  # larghezza tile per seamless scrolling
        ground_base = int(SCREEN_HEIGHT * 0.82)

        self.layers: list[ScrollingLayer] = []

        # Layer 1: Montagne lontane
        mt_far = _generate_mountains(
            tile_w, SCREEN_HEIGHT, MOUNTAIN_FAR,
            peak_min=int(SCREEN_HEIGHT * 0.30),
            peak_max=int(SCREEN_HEIGHT * 0.50),
            segments=12,
        )
        self.layers.append(ScrollingLayer(mt_far, PARALLAX_MOUNTAIN_FAR_SPEED))

        # Layer 2: Montagne medie
        mt_mid = _generate_mountains(
            tile_w, SCREEN_HEIGHT, MOUNTAIN_MID,
            peak_min=int(SCREEN_HEIGHT * 0.40),
            peak_max=int(SCREEN_HEIGHT * 0.58),
            segments=16,
        )
        self.layers.append(ScrollingLayer(mt_mid, PARALLAX_MOUNTAIN_MID_SPEED))

        # Layer 3: Montagne vicine / colline
        mt_near = _generate_mountains(
            tile_w, SCREEN_HEIGHT, MOUNTAIN_NEAR,
            peak_min=int(SCREEN_HEIGHT * 0.55),
            peak_max=int(SCREEN_HEIGHT * 0.70),
            segments=20,
        )
        self.layers.append(ScrollingLayer(mt_near, PARALLAX_MOUNTAIN_NEAR_SPEED))

        # Layer 4: Alberi
        trees = _generate_trees(
            tile_w, SCREEN_HEIGHT, ground_base,
            color_trunk=BROWN, color_leaves=DARK_GREEN,
            count=40,
        )
        self.layers.append(ScrollingLayer(trees, PARALLAX_TREES_SPEED))

        # Layer 5: Terreno
        ground_h = SCREEN_HEIGHT - ground_base + 5
        ground = _generate_ground(tile_w, ground_h, GROUND_COLOR)
        self.layers.append(ScrollingLayer(ground, PARALLAX_GROUND_SPEED, ground_base))

    def update(self, dt: float):
        for layer in self.layers:
            layer.update(dt)

    def draw(self, surface: pygame.Surface):
        # Cielo sfumato
        surface.fill(SKY_BLUE)
        gradient_rect = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT // 2), pygame.SRCALPHA)
        gradient_rect.fill((*LIGHT_BLUE, 80))
        surface.blit(gradient_rect, (0, 0))

        for layer in self.layers:
            layer.draw(surface)
