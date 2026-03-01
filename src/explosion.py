import pygame


class Explosion(pygame.sprite.Sprite):
    """Esplosione animata che si genera quando un nemico viene distrutto."""

    _frames_cache: list[pygame.Surface] | None = None

    def __init__(self, x: int, y: int):
        super().__init__()
        if Explosion._frames_cache is None:
            Explosion._frames_cache = self._generate_frames()
        self.frames = Explosion._frames_cache
        self.frame_index = 0
        self.image = self.frames[0]
        self.rect = self.image.get_rect(center=(x, y))
        self.anim_speed = 0  # contatore sub-frame

    @staticmethod
    def _generate_frames() -> list[pygame.Surface]:
        """Genera 7 frame di esplosione pixel-art."""
        PX = 3
        grid = 16
        size = grid * PX
        frames = []

        # Palette esplosione
        WHITE_HOT = (255, 255, 220)
        YELLOW_B = (255, 240, 100)
        YELLOW = (255, 200, 50)
        ORANGE = (255, 140, 30)
        RED_O = (240, 80, 15)
        RED = (200, 45, 10)
        DARK_RED = (140, 25, 5)
        SMOKE_L = (120, 100, 80, 180)
        SMOKE_D = (70, 60, 50, 140)
        SMOKE_F = (50, 45, 40, 80)

        def make_frame(pixels: list[tuple]):
            surf = pygame.Surface((size, size), pygame.SRCALPHA)
            for x, y, color in pixels:
                if 0 <= x < grid and 0 <= y < grid:
                    pygame.draw.rect(surf, color, (x * PX, y * PX, PX, PX))
            return surf

        cx, cy = 8, 8

        # Frame 0: flash iniziale piccolo (bianco/giallo)
        frames.append(make_frame([
            (cx, cy, WHITE_HOT),
            (cx - 1, cy, YELLOW_B), (cx + 1, cy, YELLOW_B),
            (cx, cy - 1, YELLOW_B), (cx, cy + 1, YELLOW_B),
        ]))

        # Frame 1: esplosione che si espande (bianco centro + giallo)
        frames.append(make_frame([
            (cx, cy, WHITE_HOT),
            (cx - 1, cy, WHITE_HOT), (cx + 1, cy, WHITE_HOT),
            (cx, cy - 1, WHITE_HOT), (cx, cy + 1, WHITE_HOT),
            (cx - 2, cy, YELLOW_B), (cx + 2, cy, YELLOW_B),
            (cx, cy - 2, YELLOW_B), (cx, cy + 2, YELLOW_B),
            (cx - 1, cy - 1, YELLOW), (cx + 1, cy - 1, YELLOW),
            (cx - 1, cy + 1, YELLOW), (cx + 1, cy + 1, YELLOW),
            (cx - 2, cy - 1, ORANGE), (cx + 2, cy + 1, ORANGE),
        ]))

        # Frame 2: massima espansione fuoco
        f2 = []
        # Centro bianco
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                f2.append((cx + dx, cy + dy, WHITE_HOT))
        # Anello giallo
        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2),
                       (-2, -1), (-2, 1), (2, -1), (2, 1),
                       (-1, -2), (1, -2), (-1, 2), (1, 2)]:
            f2.append((cx + dx, cy + dy, YELLOW))
        # Anello arancione
        for dx, dy in [(-3, 0), (3, 0), (0, -3), (0, 3),
                       (-2, -2), (2, -2), (-2, 2), (2, 2),
                       (-3, -1), (3, 1), (-1, -3), (1, 3)]:
            f2.append((cx + dx, cy + dy, ORANGE))
        # Punte rosse
        for dx, dy in [(-4, 0), (4, 0), (0, -4), (0, 4),
                       (-3, -2), (3, 2), (-3, 2), (3, -2)]:
            f2.append((cx + dx, cy + dy, RED_O))
        frames.append(make_frame(f2))

        # Frame 3: fuoco che si ritira, arancione dominante
        f3 = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                f3.append((cx + dx, cy + dy, YELLOW))
        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2),
                       (-2, -1), (2, 1), (1, -2), (-1, 2)]:
            f3.append((cx + dx, cy + dy, ORANGE))
        for dx, dy in [(-3, 0), (3, 0), (0, -3), (0, 3),
                       (-3, -1), (3, 1), (-2, -2), (2, 2),
                       (-2, 2), (2, -2)]:
            f3.append((cx + dx, cy + dy, RED_O))
        for dx, dy in [(-4, 0), (4, 1), (0, -4), (1, 4),
                       (-4, -1), (3, -2)]:
            f3.append((cx + dx, cy + dy, RED))
        frames.append(make_frame(f3))

        # Frame 4: rosso/fumo, fuoco si spegne
        f4 = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                f4.append((cx + dx, cy + dy, ORANGE))
        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2),
                       (-2, -1), (2, 1)]:
            f4.append((cx + dx, cy + dy, RED_O))
        for dx, dy in [(-3, 0), (3, 0), (0, -3), (0, 3),
                       (-3, 1), (3, -1), (-2, -2), (2, 2)]:
            f4.append((cx + dx, cy + dy, DARK_RED))
        for dx, dy in [(-4, -1), (4, 1), (-1, -4), (1, 4),
                       (-3, -2), (3, 2)]:
            f4.append((cx + dx, cy + dy, SMOKE_L))
        frames.append(make_frame(f4))

        # Frame 5: fumo grigio chiaro
        f5 = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                f5.append((cx + dx, cy + dy, DARK_RED))
        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2),
                       (-2, 1), (2, -1), (-1, -2), (1, 2)]:
            f5.append((cx + dx, cy + dy, SMOKE_L))
        for dx, dy in [(-3, 0), (3, 1), (0, -3), (1, 3),
                       (-3, -1), (2, -2), (-2, 2)]:
            f5.append((cx + dx, cy + dy, SMOKE_D))
        frames.append(make_frame(f5))

        # Frame 6: fumo che si dissolve
        f6 = []
        for dx, dy in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            f6.append((cx + dx, cy + dy, SMOKE_D))
        for dx, dy in [(-2, 0), (2, 1), (0, -2), (1, 2),
                       (-1, -1), (1, 1)]:
            f6.append((cx + dx, cy + dy, SMOKE_F))
        frames.append(make_frame(f6))

        return frames

    def update(self, dt: float):
        self.anim_speed += 1
        if self.anim_speed >= 3:  # cambia frame ogni 3 tick
            self.anim_speed = 0
            self.frame_index += 1
            if self.frame_index >= len(self.frames):
                self.kill()
            else:
                self.image = self.frames[self.frame_index]
