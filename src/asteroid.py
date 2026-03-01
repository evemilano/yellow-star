import random
import math
import pygame
from settings import (
    SCREEN_WIDTH, SCREEN_HEIGHT,
    ASTEROID_SPEED_MIN, ASTEROID_SPEED_MAX,
    ASTEROID_SIZE_MIN, ASTEROID_SIZE_MAX, ASTEROID_COLOR,
)


class Asteroid(pygame.sprite.Sprite):
    def __init__(self, speed_min=None, speed_max=None,
                 size_min=None, size_max=None):
        super().__init__()
        sz_min = size_min if size_min is not None else ASTEROID_SIZE_MIN
        sz_max = size_max if size_max is not None else ASTEROID_SIZE_MAX
        self.radius = random.randint(sz_min, sz_max)
        self.image = self._create_sprite()
        self.rect = self.image.get_rect(
            center=(
                SCREEN_WIDTH + self.radius * 2,
                random.randint(self.radius, SCREEN_HEIGHT - self.radius),
            )
        )
        smin = speed_min if speed_min is not None else ASTEROID_SPEED_MIN
        smax = speed_max if speed_max is not None else ASTEROID_SPEED_MAX
        self.speed = random.uniform(smin, smax)
        self.angle = 0
        self.rotation_speed = random.uniform(-2, 2)

    def _create_sprite(self) -> pygame.Surface:
        """Asteroide irregolare generato proceduralmente."""
        size = self.radius * 2 + 4
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        cx, cy = size // 2, size // 2

        # Genera un poligono irregolare
        num_points = random.randint(7, 10)
        points = []
        for i in range(num_points):
            angle = (2 * math.pi / num_points) * i
            r = self.radius * random.uniform(0.7, 1.0)
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            points.append((x, y))

        pygame.draw.polygon(surf, ASTEROID_COLOR, points)
        # Bordo più scuro
        pygame.draw.polygon(surf, (100, 85, 70), points, 2)
        # Crateri
        for _ in range(random.randint(1, 3)):
            cr = random.randint(2, max(3, self.radius // 5))
            cx2 = cx + random.randint(-self.radius // 2, self.radius // 2)
            cy2 = cy + random.randint(-self.radius // 2, self.radius // 2)
            pygame.draw.circle(surf, (100, 85, 70), (cx2, cy2), cr)

        return surf

    def update(self, dt: float):
        self.rect.x -= int(self.speed * dt)
        if self.rect.right < 0:
            self.kill()
