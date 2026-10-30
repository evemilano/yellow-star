import math
import random
import threading
import pygame
from settings import (
    SCREEN_WIDTH, SCREEN_HEIGHT,
    PARALLAX_MOUNTAIN_FAR_SPEED, PARALLAX_MOUNTAIN_MID_SPEED,
    PARALLAX_MOUNTAIN_NEAR_SPEED, PARALLAX_TREES_SPEED, PARALLAX_GROUND_SPEED,
    BIOME_DURATION, BIOME_FADE_DURATION, BIOME_SPACE_LEVEL,
)


# ── Utility di disegno ──

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _mix(c1: tuple, c2: tuple, t: float) -> tuple:
    """Interpola due colori RGB."""
    return (
        int(_lerp(c1[0], c2[0], t)),
        int(_lerp(c1[1], c2[1], t)),
        int(_lerp(c1[2], c2[2], t)),
    )


def _shade(color: tuple, factor: float) -> tuple:
    """Schiarisce (factor>1) o scurisce (factor<1) un colore."""
    return (
        max(0, min(255, int(color[0] * factor))),
        max(0, min(255, int(color[1] * factor))),
        max(0, min(255, int(color[2] * factor))),
    )


def _vertical_gradient(width: int, height: int, top: tuple, bottom: tuple,
                       bands: int = 64) -> pygame.Surface:
    """Gradiente verticale a bande (molto piu' veloce di pixel-per-pixel)."""
    surf = pygame.Surface((width, height))
    band_h = max(1, height // bands)
    y = 0
    while y < height:
        t = y / max(1, height - 1)
        pygame.draw.rect(surf, _mix(top, bottom, t), (0, y, width, band_h + 1))
        y += band_h
    return surf


def _ridge_points(width: int, segments: int, top: int, bottom: int,
                  roughness: float, rng: random.Random) -> list:
    """Genera una cresta frastagliata via midpoint displacement.

    Ritorna `segments+1` altezze y, con dettaglio ricorsivo: le montagne
    hanno picchi principali e micro-irregolarita' invece di zig-zag uniforme.
    """
    # Numero di punti = potenza di 2 + 1, per il midpoint displacement
    n = 1
    while n < segments:
        n *= 2
    pts = [0.0] * (n + 1)
    pts[0] = rng.uniform(top, bottom)
    pts[n] = rng.uniform(top, bottom)

    step = n
    scale = (bottom - top) * roughness
    while step > 1:
        half = step // 2
        for i in range(half, n, step):
            avg = (pts[i - half] + pts[i + half]) / 2.0
            pts[i] = avg + rng.uniform(-scale, scale)
        step = half
        scale *= 0.55
    return [max(top - 20, min(bottom + 20, p)) for p in pts]


def _draw_ridge(surf: pygame.Surface, width: int, height: int,
                heights: list, color: tuple, snow: tuple = None,
                snow_line: float = None, haze: tuple = None):
    """Disegna una catena montuosa da un profilo di altezze."""
    n = len(heights) - 1
    step = width / n
    poly = [(0, height)]
    for i, y in enumerate(heights):
        poly.append((i * step, y))
    poly.append((width, height))
    pygame.draw.polygon(surf, color, poly)

    # Ombreggiatura dei versanti: si accumulano i segmenti consecutivi che
    # guardano nella stessa direzione e si riempie l'intero versante fino a
    # valle in un colpo solo. Ombreggiare segmento per segmento produce un
    # motivo a mattoncini lungo il crinale.
    light = _shade(color, 1.14)
    dark = _shade(color, 0.86)
    i = 0
    while i < n:
        rising = heights[i + 1] < heights[i]
        j = i
        while j < n and (heights[j + 1] < heights[j]) == rising:
            j += 1
        # Quota di valle del versante: il piu' basso dei due estremi
        foot = max(heights[i], heights[j])
        ridge = [((i + k) * step, heights[i + k]) for k in range(j - i + 1)]
        pygame.draw.polygon(surf, light if rising else dark,
                            ridge + [(j * step, foot), (i * step, foot)])
        i = j

    # Cappelli di neve: il triangolo di vetta SOPRA la snow_line
    if snow and snow_line is not None:
        for i in range(n):
            y0, y1 = heights[i], heights[i + 1]
            if y0 >= snow_line and y1 >= snow_line:
                continue  # segmento interamente sotto quota neve
            x0, x1 = i * step, (i + 1) * step
            pts = []
            if y0 < snow_line:
                pts.append((x0, y0))
            # Intersezioni del segmento con la linea della neve
            if (y0 < snow_line) != (y1 < snow_line):
                t = (snow_line - y0) / (y1 - y0)
                pts.append((x0 + (x1 - x0) * t, snow_line))
            if y1 < snow_line:
                pts.append((x1, y1))
            if y0 < snow_line and y1 < snow_line:
                pts.extend([(x1, snow_line), (x0, snow_line)])
            if len(pts) >= 3:
                pygame.draw.polygon(surf, snow, pts)

    # Foschia atmosferica alla base: fonde la montagna col cielo
    if haze:
        band = pygame.Surface((width, height // 5), pygame.SRCALPHA)
        h = band.get_height()
        for y in range(0, h, 3):
            alpha = int(110 * (y / h))
            pygame.draw.rect(band, (*haze, alpha), (0, y, width, 3))
        surf.blit(band, (0, height - h))


def _draw_star_field(surf: pygame.Surface, width: int, height: int,
                     count: int, rng: random.Random, max_r: int = 2):
    """Stelle con dimensione e luminosita' variabili."""
    for _ in range(count):
        x = rng.randint(0, width)
        y = rng.randint(0, height)
        r = rng.randint(1, max_r)
        b = rng.randint(150, 255)
        tint = rng.choice([(b, b, b), (b, b, 255), (255, b, b - 30)])
        pygame.draw.circle(surf, tint, (x, y), r)
        if r > 1:  # alone per le stelle piu' luminose
            pygame.draw.circle(surf, (*tint, 60), (x, y), r + 2, 1)


def _alpha_rows(surface: pygame.Surface):
    """Prima riga e altezza della fascia con pixel non trasparenti (alpha > 0).

    Stesse righe di get_bounding_rect(min_alpha=1), ma calcolate via mask:
    get_bounding_rect scandisce i pixel uno a uno e costa 4-6 volte tanto.
    Ritorna None se la superficie e' tutta trasparente.
    """
    mask = pygame.mask.from_surface(surface, 0)  # bit = alpha > 0
    w, h = mask.get_size()
    row = pygame.mask.Mask((w, 1), fill=True)
    top = 0
    while top < h and not mask.overlap_area(row, (0, top)):
        top += 1
    if top == h:
        return None
    bottom = h - 1
    while not mask.overlap_area(row, (0, bottom)):
        bottom -= 1
    return top, bottom - top + 1


# ── Layer scorrevole ──

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


# ── Biomi ──

class Biome:
    """Uno scenario completo: cielo + N layer scorrevoli.

    Ogni sottoclasse implementa `build()` popolando `self.layers` e
    impostando `self.sky`.
    """

    name = "biome"
    crop_layers = True  # ritaglio dei layer alla fascia non trasparente (add)

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.layers: list[ScrollingLayer] = []
        self.sky: pygame.Surface | None = None
        self.tile_w = SCREEN_WIDTH * 2
        self.ground_y = int(SCREEN_HEIGHT * 0.82)
        self.build()

    def build(self):
        raise NotImplementedError

    def add(self, surface: pygame.Surface, speed: float, y_offset: int = 0):
        # Ritaglia le righe interamente trasparenti sopra e sotto il contenuto:
        # molti layer occupano solo una fascia e il blit con alpha per-pixel
        # costa per tutta l'altezza. I pixel con alpha 0 non cambiano la
        # destinazione, quindi l'immagine resta identica; la larghezza (tile
        # seamless) non cambia.
        if self.crop_layers and surface.get_flags() & pygame.SRCALPHA:
            rows = _alpha_rows(surface)
            if rows is not None and rows[1] < surface.get_height():
                top, height = rows
                surface = surface.subsurface((0, top, surface.get_width(), height)).copy()
                y_offset += top
        self.layers.append(ScrollingLayer(surface, speed, y_offset))

    def update(self, dt: float):
        for layer in self.layers:
            layer.update(dt)

    def draw(self, surface: pygame.Surface):
        if self.sky:
            surface.blit(self.sky, (0, 0))
        for layer in self.layers:
            layer.draw(surface)


class MountainBiome(Biome):
    """Catene montuose innevate con foresta di conifere."""

    name = "montagna"

    def build(self):
        rng = self.rng
        w, h = self.tile_w, SCREEN_HEIGHT
        sky_top = rng.choice([(70, 120, 190), (90, 140, 200), (120, 150, 200)])
        self.sky = _vertical_gradient(SCREEN_WIDTH, h, sky_top, (205, 225, 240))

        # Sole con alone
        sun_x = rng.randint(200, SCREEN_WIDTH - 200)
        sun_y = rng.randint(60, 160)
        glow = pygame.Surface((300, 300), pygame.SRCALPHA)
        for r in range(140, 0, -6):
            a = int(50 * (1 - r / 140))
            pygame.draw.circle(glow, (255, 245, 200, a), (150, 150), r)
        self.sky.blit(glow, (sun_x - 150, sun_y - 150))
        pygame.draw.circle(self.sky, (255, 250, 225), (sun_x, sun_y), 34)

        # Tre creste, dalla piu' lontana alla piu' vicina
        ridges = [
            ((0.26, 0.48), (86, 104, 134), 64, 0.75, PARALLAX_MOUNTAIN_FAR_SPEED),
            ((0.38, 0.60), (62, 82, 108), 96, 0.68, PARALLAX_MOUNTAIN_MID_SPEED),
            ((0.52, 0.72), (40, 58, 80), 128, 0.60, PARALLAX_MOUNTAIN_NEAR_SPEED),
        ]
        for (lo, hi), color, seg, rough, speed in ridges:
            surf = pygame.Surface((w, h), pygame.SRCALPHA)
            hs = _ridge_points(w, seg, int(h * lo), int(h * hi), rough, rng)
            # snow_line poco sotto la quota minima dei picchi: solo le vette
            # piu' alte la superano, quindi la neve resta sporadica
            _draw_ridge(surf, w, h, hs, color,
                        snow=_mix(color, (255, 255, 255), 0.82),
                        snow_line=h * lo + (h * (hi - lo)) * 0.30,
                        haze=(180, 200, 225))
            self.add(surf, speed)

        # Conifere davanti
        trees = pygame.Surface((w, h), pygame.SRCALPHA)
        for _ in range(70):
            x = rng.randint(0, w)
            th = rng.randint(50, 110)
            tw = th // 3
            base = self.ground_y + rng.randint(-8, 6)
            green = (rng.randint(14, 32), rng.randint(52, 88), rng.randint(24, 44))
            pygame.draw.rect(trees, (58, 38, 22), (x - 2, base - th // 4, 4, th // 4))
            for tier in range(3):  # chioma a tre piani
                ty = base - th + tier * (th // 3)
                tsw = tw + tier * (tw // 3)
                pygame.draw.polygon(trees, green, [
                    (x, ty), (x - tsw, ty + th // 2), (x + tsw, ty + th // 2)
                ])
        self.add(trees, PARALLAX_TREES_SPEED)

        self.add(self._rocky_ground(rng), PARALLAX_GROUND_SPEED, self.ground_y)

    def _rocky_ground(self, rng):
        w = self.tile_w
        gh = SCREEN_HEIGHT - self.ground_y + 5
        surf = pygame.Surface((w, gh), pygame.SRCALPHA)
        pts = [(0, gh)] + [(i * w // 40, rng.randint(0, 14)) for i in range(41)] + [(w, gh)]
        pygame.draw.polygon(surf, (58, 92, 48), pts)
        for _ in range(120):  # sassi e ciuffi d'erba
            x, y = rng.randint(0, w), rng.randint(12, gh)
            pygame.draw.circle(surf, _shade((58, 92, 48), rng.uniform(0.7, 1.3)),
                               (x, y), rng.randint(2, 6))
        return surf


class SeaBiome(Biome):
    """Oceano con isole, onde e riflesso del sole."""

    name = "mare"

    def build(self):
        rng = self.rng
        w, h = self.tile_w, SCREEN_HEIGHT
        horizon = int(h * 0.62)
        self.sky = _vertical_gradient(SCREEN_WIDTH, h, (60, 130, 205), (225, 210, 175))

        sun_x = rng.randint(300, SCREEN_WIDTH - 300)
        pygame.draw.circle(self.sky, (255, 235, 190), (sun_x, horizon - 90), 42)

        # Nuvole stratificate lontane
        clouds = pygame.Surface((w, h), pygame.SRCALPHA)
        for _ in range(28):
            cx, cy = rng.randint(0, w), rng.randint(40, horizon - 130)
            cw = rng.randint(90, 240)
            for _ in range(rng.randint(3, 6)):
                pygame.draw.ellipse(clouds, (255, 255, 255, rng.randint(40, 90)), (
                    cx + rng.randint(-cw // 2, cw // 2), cy + rng.randint(-14, 14),
                    rng.randint(60, cw), rng.randint(18, 40)))
        self.add(clouds, PARALLAX_MOUNTAIN_FAR_SPEED)

        # Isole all'orizzonte
        islands = pygame.Surface((w, h), pygame.SRCALPHA)
        for _ in range(7):
            ix = rng.randint(0, w)
            iw, ih = rng.randint(140, 320), rng.randint(40, 110)
            col = (rng.randint(46, 70), rng.randint(74, 100), rng.randint(60, 82))
            pygame.draw.polygon(islands, col, [
                (ix - iw // 2, horizon), (ix - iw // 5, horizon - ih),
                (ix + iw // 8, horizon - ih * 0.7), (ix + iw // 2, horizon)])
            pygame.draw.polygon(islands, _shade(col, 1.25), [
                (ix - iw // 5, horizon - ih), (ix, horizon - ih * 0.55),
                (ix - iw // 3, horizon)])
        self.add(islands, PARALLAX_MOUNTAIN_MID_SPEED)

        # Acqua: gradiente + bande d'onda a velocita' diverse
        sea_h = h - horizon
        water = pygame.Surface((w, sea_h), pygame.SRCALPHA)
        water.blit(_vertical_gradient(w, sea_h, (28, 92, 150), (16, 52, 96)), (0, 0))
        # Riflesso solare: frammenti sparsi dentro un cono che si allarga.
        # Un triangolo pieno sembrerebbe una passerella, non acqua.
        for y in range(0, sea_h, 5):
            spread = 30 + int(y * 1.6)
            a = max(0, 110 - int(y * 0.42))
            if a <= 0:
                continue
            for _ in range(rng.randint(1, 4)):
                seg = rng.randint(6, max(8, spread // 3))
                sx2 = sun_x + rng.randint(-spread // 2, spread // 2) - seg // 2
                pygame.draw.rect(water, (255, 240, 200, rng.randint(a // 3, a)),
                                 (sx2, y + rng.randint(0, 2), seg, rng.randint(2, 3)))
        self.add(water, PARALLAX_MOUNTAIN_NEAR_SPEED, horizon)

        for idx, (speed, cnt, alpha) in enumerate((
                (PARALLAX_TREES_SPEED, 110, 70), (PARALLAX_GROUND_SPEED, 70, 130))):
            waves = pygame.Surface((w, sea_h), pygame.SRCALPHA)
            for _ in range(cnt):
                y = rng.randint(int(sea_h * (0.15 + idx * 0.35)), sea_h - 4)
                ln = rng.randint(30, 130) + idx * 40
                x = rng.randint(0, w)
                pygame.draw.line(waves, (235, 248, 255, alpha), (x, y), (x + ln, y),
                                 1 + idx)
            self.add(waves, speed, horizon)


class CityBiome(Biome):
    """Skyline notturno con monumenti famosi, finestre accese e riflessi."""

    name = "citta"

    def build(self):
        rng = self.rng
        w, h = self.tile_w, SCREEN_HEIGHT
        self.sky = _vertical_gradient(SCREEN_WIDTH, h, (14, 16, 46), (196, 96, 92))
        _draw_star_field(self.sky, SCREEN_WIDTH, int(h * 0.45), 120, rng, max_r=1)
        # Luna
        mx, my = rng.randint(150, SCREEN_WIDTH - 150), rng.randint(50, 130)
        pygame.draw.circle(self.sky, (245, 242, 220), (mx, my), 30)
        pygame.draw.circle(self.sky, (225, 222, 200), (mx - 8, my - 5), 6)
        pygame.draw.circle(self.sky, (225, 222, 200), (mx + 9, my + 8), 4)

        # Tre fasce di skyline, sempre piu' scure e vicine
        bands = [
            ((24, 26, 58), 0.30, 0.50, 46, PARALLAX_MOUNTAIN_FAR_SPEED, 0.25),
            ((16, 18, 44), 0.26, 0.44, 30, PARALLAX_MOUNTAIN_MID_SPEED, 0.5),
            # Fascia vicina bassa e diradata: se e' troppo alta nasconde i monumenti
            ((8, 9, 26), 0.14, 0.30, 20, PARALLAX_MOUNTAIN_NEAR_SPEED, 0.85),
        ]
        for color, lo, hi, count, speed, lit in bands:
            surf = pygame.Surface((w, h), pygame.SRCALPHA)
            x = 0
            while x < w:
                bw = rng.randint(w // (count * 2), w // count)
                bh = rng.randint(int(h * lo), int(h * hi))
                top = self.ground_y - bh
                pygame.draw.rect(surf, color, (x, top, bw, bh))
                pygame.draw.rect(surf, _shade(color, 1.5), (x, top, bw, bh), 1)
                # Antenna occasionale
                if rng.random() < 0.25:
                    pygame.draw.rect(surf, color, (x + bw // 2 - 1, top - 30, 3, 30))
                    pygame.draw.circle(surf, (255, 70, 70), (x + bw // 2, top - 32), 3)
                # Finestre accese
                for wy in range(top + 8, self.ground_y - 8, 14):
                    for wx in range(x + 5, x + bw - 6, 11):
                        if rng.random() < lit * 0.55:
                            glow = rng.choice([(255, 224, 150), (255, 196, 110),
                                               (180, 220, 255)])
                            pygame.draw.rect(surf, glow, (wx, wy, 5, 8))
                x += bw + rng.randint(2, 14)
            self.add(surf, speed)

        # Monumenti su un layer proprio, davanti agli edifici: se disegnati
        # dentro una fascia verrebbero coperti da quella successiva
        marks = pygame.Surface((w, h), pygame.SRCALPHA)
        self._add_landmarks(marks, rng)
        self._outline_landmarks(marks, w, h)
        self.add(marks, PARALLAX_MOUNTAIN_NEAR_SPEED)

        # Strada in primo piano con luci
        road_h = SCREEN_HEIGHT - self.ground_y + 5
        road = pygame.Surface((w, road_h), pygame.SRCALPHA)
        road.fill((18, 18, 28))
        for x in range(0, w, 60):
            pygame.draw.rect(road, (90, 90, 110), (x, road_h // 2, 34, 3))
        for x in range(0, w, 220):  # lampioni
            pygame.draw.rect(road, (40, 40, 55), (x, 0, 4, road_h // 2))
            pygame.draw.circle(road, (255, 220, 140), (x + 2, 4), 6)
        self.add(road, PARALLAX_GROUND_SPEED, self.ground_y)

    def _outline_landmarks(self, surf, w, h):
        """Alone caldo attorno alle sagome, per staccarle dallo skyline."""
        mask = pygame.mask.from_surface(surf)
        glow = pygame.Surface((w, h), pygame.SRCALPHA)
        silhouette = mask.to_surface(
            setcolor=(255, 186, 120, 70), unsetcolor=(0, 0, 0, 0))
        for dx, dy in ((-3, 0), (3, 0), (0, -3), (0, 3), (-2, -2), (2, 2)):
            glow.blit(silhouette, (dx, dy))
        glow.blit(surf, (0, 0))
        surf.blit(glow, (0, 0))

    def _add_landmarks(self, surf, rng):
        """Sagome di monumenti famosi, distribuite senza sovrapporsi."""
        builders = [self._eiffel, self._big_ben, self._pyramid,
                    self._pagoda, self._colosseum, self._empire]
        rng.shuffle(builders)
        slot = self.tile_w // len(builders)
        for i, build in enumerate(builders):
            x = i * slot + rng.randint(slot // 4, slot * 3 // 4)
            build(surf, x, self.ground_y, rng)

    def _eiffel(self, s, x, gy, rng):
        c, hgt = (46, 48, 82), rng.randint(210, 300)
        top = gy - hgt
        pygame.draw.polygon(s, c, [(x - hgt // 4, gy), (x - hgt // 20, top),
                                   (x + hgt // 20, top), (x + hgt // 4, gy)])
        pygame.draw.polygon(s, (0, 0, 0, 0), [
            (x - hgt // 6, gy), (x - hgt // 22, gy - hgt // 2),
            (x + hgt // 22, gy - hgt // 2), (x + hgt // 6, gy)])
        pygame.draw.rect(s, c, (x - hgt // 5, gy - hgt // 2.6, hgt // 2.5, 7))
        pygame.draw.rect(s, c, (x - hgt // 9, gy - hgt // 1.5, hgt // 4.5, 5))
        pygame.draw.rect(s, c, (x - 2, top - 22, 4, 22))
        pygame.draw.circle(s, (255, 210, 120), (x, top - 24), 3)

    def _big_ben(self, s, x, gy, rng):
        c, hgt = (50, 46, 78), rng.randint(180, 250)
        top = gy - hgt
        pygame.draw.rect(s, c, (x - 22, top + 40, 44, hgt - 40))
        pygame.draw.polygon(s, c, [(x - 26, top + 44), (x, top), (x + 26, top + 44)])
        pygame.draw.circle(s, (255, 236, 178), (x, top + 78), 15)
        pygame.draw.circle(s, (60, 50, 30), (x, top + 78), 15, 2)
        pygame.draw.line(s, (40, 34, 20), (x, top + 78), (x, top + 68), 2)
        pygame.draw.line(s, (40, 34, 20), (x, top + 78), (x + 8, top + 78), 2)

    def _pyramid(self, s, x, gy, rng):
        c, hgt = (66, 56, 74), rng.randint(120, 190)
        pygame.draw.polygon(s, c, [(x - hgt, gy), (x, gy - hgt), (x + hgt, gy)])
        pygame.draw.polygon(s, _shade(c, 1.4), [(x, gy - hgt), (x + hgt, gy), (x, gy)])

    def _pagoda(self, s, x, gy, rng):
        c, tiers = (54, 44, 72), rng.randint(4, 6)
        th = 34
        for i in range(tiers):
            y = gy - (i + 1) * th
            hw = 58 - i * 8
            pygame.draw.rect(s, c, (x - hw + 8, y, (hw - 8) * 2, th))
            pygame.draw.polygon(s, _shade(c, 1.6), [
                (x - hw, y + 10), (x, y - 8), (x + hw, y + 10)])
        pygame.draw.rect(s, c, (x - 2, gy - tiers * th - 26, 4, 22))

    def _colosseum(self, s, x, gy, rng):
        c, hgt, wd = (58, 52, 80), rng.randint(90, 130), rng.randint(150, 210)
        pygame.draw.ellipse(s, c, (x - wd // 2, gy - hgt, wd, hgt * 2))
        pygame.draw.rect(s, (0, 0, 0, 0), (x - wd // 2, gy, wd, hgt))
        for row in range(3):
            ay = gy - hgt + 14 + row * (hgt // 3)
            for ax in range(x - wd // 2 + 12, x + wd // 2 - 10, 20):
                pygame.draw.circle(s, _shade(c, 2.2), (ax, ay), 5)

    def _empire(self, s, x, gy, rng):
        c, hgt = (48, 50, 84), rng.randint(230, 320)
        top = gy - hgt
        # Tre corpi rastremati impilati: base larga, poi via via piu' stretti
        tiers = ((46, 0.45), (32, 0.32), (20, 0.23))
        y = gy
        for fw, frac in tiers:
            seg = hgt * frac
            y -= seg
            pygame.draw.rect(s, c, (x - fw, y, fw * 2, seg + 2))
        pygame.draw.rect(s, c, (x - 3, top - 40, 6, 44))
        pygame.draw.circle(s, (255, 200, 120), (x, top - 42), 3)


class JungleBiome(Biome):
    """Foresta tropicale: nebbia, palme, felci e liane."""

    name = "foresta"

    def build(self):
        rng = self.rng
        w, h = self.tile_w, SCREEN_HEIGHT
        self.sky = _vertical_gradient(SCREEN_WIDTH, h, (120, 190, 170), (232, 226, 178))

        # Raggi di luce che filtrano dall'alto
        rays = pygame.Surface((SCREEN_WIDTH, h), pygame.SRCALPHA)
        for _ in range(9):
            rx = rng.randint(0, SCREEN_WIDTH)
            pygame.draw.polygon(rays, (255, 250, 205, 26), [
                (rx, 0), (rx + 60, 0), (rx + 190, h), (rx + 40, h)])
        self.sky.blit(rays, (0, 0))

        # Colline nebbiose sul fondo
        for i, (lo, hi, col, speed) in enumerate((
                (0.34, 0.52, (104, 150, 124), PARALLAX_MOUNTAIN_FAR_SPEED),
                (0.46, 0.64, (72, 118, 92), PARALLAX_MOUNTAIN_MID_SPEED))):
            surf = pygame.Surface((w, h), pygame.SRCALPHA)
            hs = _ridge_points(w, 24, int(h * lo), int(h * hi), 0.5, rng)
            _draw_ridge(surf, w, h, hs, col, haze=(200, 220, 200))
            self.add(surf, speed)

        # Canopy: massa di chiome scure
        canopy = pygame.Surface((w, h), pygame.SRCALPHA)
        for _ in range(190):
            cx = rng.randint(0, w)
            cy = rng.randint(int(h * 0.48), int(h * 0.74))
            r = rng.randint(26, 62)
            g = (rng.randint(18, 42), rng.randint(62, 104), rng.randint(28, 56))
            pygame.draw.circle(canopy, g, (cx, cy), r)
            pygame.draw.circle(canopy, _shade(g, 1.3), (cx - r // 4, cy - r // 4),
                               r // 2)
        self.add(canopy, PARALLAX_MOUNTAIN_NEAR_SPEED)

        # Palme in primo piano
        palms = pygame.Surface((w, h), pygame.SRCALPHA)
        for _ in range(34):
            x = rng.randint(0, w)
            th = rng.randint(120, 240)
            base = self.ground_y + rng.randint(-6, 8)
            lean = rng.randint(-26, 26)
            top_x, top_y = x + lean, base - th
            # Tronco: molti cerchi ravvicinati formano una curva continua che
            # si assottiglia verso l'alto. Con pochi passi restano pallini staccati.
            steps = max(24, th // 5)
            for k in range(steps + 1):
                t = k / steps
                pygame.draw.circle(palms, _mix((78, 56, 32), (104, 78, 46), t),
                                   (int(_lerp(x, top_x, t)), int(_lerp(base, top_y, t))),
                                   max(3, int(_lerp(9, 4, t))))
            for a in range(7):  # fronde
                ang = math.pi + a * (math.pi / 6) + rng.uniform(-0.15, 0.15)
                ln = rng.randint(50, 95)
                ex = top_x + math.cos(ang) * ln
                ey = top_y - math.sin(ang) * ln * 0.6
                col = (rng.randint(24, 48), rng.randint(96, 140), rng.randint(38, 62))
                pygame.draw.polygon(palms, col, [
                    (top_x, top_y), (ex, ey),
                    (ex + rng.randint(-14, 14), ey + rng.randint(10, 24))])
        # Liane: partono dalla fascia delle chiome, non dal bordo dello schermo,
        # altrimenti sembrano fili sospesi nel cielo vuoto
        for _ in range(26):
            vx = rng.randint(0, w)
            vy = rng.randint(int(h * 0.46), int(h * 0.60))
            vl = rng.randint(80, 220)
            sway = rng.randint(-16, 16)
            pygame.draw.line(palms, (34, 80, 44), (vx, vy), (vx + sway, vy + vl), 3)
            pygame.draw.circle(palms, (46, 104, 56), (vx + sway, vy + vl), 5)
        self.add(palms, PARALLAX_TREES_SPEED)

        # Sottobosco
        gh = SCREEN_HEIGHT - self.ground_y + 5
        floor = pygame.Surface((w, gh), pygame.SRCALPHA)
        floor.fill((34, 66, 36))
        for _ in range(320):
            x, y = rng.randint(0, w), rng.randint(0, gh)
            col = (rng.randint(26, 70), rng.randint(80, 140), rng.randint(30, 70))
            pygame.draw.ellipse(floor, col, (x, y, rng.randint(8, 26), rng.randint(4, 10)))
        self.add(floor, PARALLAX_GROUND_SPEED, self.ground_y)


class DesertBiome(Biome):
    """Deserto al tramonto: dune, mesa e cactus."""

    name = "deserto"

    def build(self):
        rng = self.rng
        w, h = self.tile_w, SCREEN_HEIGHT
        self.sky = _vertical_gradient(SCREEN_WIDTH, h, (232, 128, 62), (250, 216, 150))
        sx, sy = rng.randint(200, SCREEN_WIDTH - 200), int(h * 0.42)
        for r in range(120, 0, -5):
            pygame.draw.circle(self.sky, (255, 190, 110), (sx, sy), r) if r < 60 else \
                pygame.draw.circle(self.sky, _mix((250, 200, 130), (255, 240, 190),
                                                  1 - r / 120), (sx, sy), r)

        # Mesa lontane
        mesa = pygame.Surface((w, h), pygame.SRCALPHA)
        for _ in range(9):
            mx = rng.randint(0, w)
            mw, mh = rng.randint(150, 340), rng.randint(70, 170)
            top = int(h * 0.56) - mh
            col = (rng.randint(130, 160), rng.randint(76, 100), rng.randint(58, 76))
            pygame.draw.polygon(mesa, col, [
                (mx - mw // 2, int(h * 0.56)), (mx - mw // 2 + 16, top),
                (mx + mw // 2 - 16, top), (mx + mw // 2, int(h * 0.56))])
            pygame.draw.rect(mesa, _shade(col, 0.8), (mx, top, mw // 2 - 16, mh))
        self.add(mesa, PARALLAX_MOUNTAIN_FAR_SPEED)

        # Dune sovrapposte
        for lo, hi, col, speed in ((0.58, 0.70, (226, 186, 122), PARALLAX_MOUNTAIN_MID_SPEED),
                                   (0.66, 0.78, (206, 162, 100), PARALLAX_MOUNTAIN_NEAR_SPEED)):
            surf = pygame.Surface((w, h), pygame.SRCALPHA)
            hs = _ridge_points(w, 16, int(h * lo), int(h * hi), 0.35, rng)
            _draw_ridge(surf, w, h, hs, col)
            self.add(surf, speed)

        # Cactus e rocce
        cacti = pygame.Surface((w, h), pygame.SRCALPHA)
        for _ in range(30):
            x = rng.randint(0, w)
            ch = rng.randint(50, 110)
            base = self.ground_y + rng.randint(-8, 4)
            col = (rng.randint(48, 76), rng.randint(100, 138), rng.randint(52, 74))
            pygame.draw.rect(cacti, col, (x - 7, base - ch, 14, ch), border_radius=7)
            for side in (-1, 1):
                if rng.random() < 0.6:
                    ay = base - rng.randint(ch // 2, int(ch * 0.8))
                    aw = rng.randint(18, 30)
                    pygame.draw.rect(cacti, col, (x + side * 7 - (aw if side < 0 else 0),
                                                  ay, aw, 10), border_radius=5)
                    pygame.draw.rect(cacti, col, (x + side * (7 + aw) - (5 if side < 0 else 0),
                                                  ay - 26, 10, 32), border_radius=5)
        self.add(cacti, PARALLAX_TREES_SPEED)

        gh = SCREEN_HEIGHT - self.ground_y + 5
        sand = pygame.Surface((w, gh), pygame.SRCALPHA)
        sand.fill((214, 172, 108))
        for _ in range(260):
            x, y = rng.randint(0, w), rng.randint(0, gh)
            pygame.draw.ellipse(sand, _shade((214, 172, 108), rng.uniform(0.85, 1.15)),
                                (x, y, rng.randint(14, 46), rng.randint(2, 5)))
        self.add(sand, PARALLAX_GROUND_SPEED, self.ground_y)


class ArcticBiome(Biome):
    """Distesa polare con aurora, iceberg e ghiaccio."""

    name = "artico"

    def build(self):
        rng = self.rng
        w, h = self.tile_w, SCREEN_HEIGHT
        self.sky = _vertical_gradient(SCREEN_WIDTH, h, (18, 30, 70), (176, 208, 232))
        _draw_star_field(self.sky, SCREEN_WIDTH, int(h * 0.35), 90, rng, max_r=1)

        # Aurora boreale
        aurora = pygame.Surface((SCREEN_WIDTH, h), pygame.SRCALPHA)
        for band in range(5):
            base_y = 60 + band * 30
            col = rng.choice([(90, 255, 170), (110, 200, 255), (180, 130, 255)])
            for x in range(0, SCREEN_WIDTH, 4):
                amp = 34 + band * 9
                y = base_y + math.sin(x * 0.006 + band) * amp
                hgt = 90 + math.sin(x * 0.013 + band * 2) * 40
                for k in range(0, int(hgt), 5):
                    a = int(46 * (1 - k / hgt))
                    pygame.draw.rect(aurora, (*col, a), (x, y + k, 4, 5))
        self.sky.blit(aurora, (0, 0))

        # Montagne di ghiaccio
        for lo, hi, col, speed in ((0.30, 0.50, (118, 148, 182), PARALLAX_MOUNTAIN_FAR_SPEED),
                                   (0.44, 0.64, (88, 120, 158), PARALLAX_MOUNTAIN_MID_SPEED)):
            surf = pygame.Surface((w, h), pygame.SRCALPHA)
            hs = _ridge_points(w, 22, int(h * lo), int(h * hi), 0.6, rng)
            _draw_ridge(surf, w, h, hs, col, snow=(238, 246, 255),
                        snow_line=h * (lo + 0.04), haze=(210, 228, 245))
            self.add(surf, speed)

        # Iceberg frastagliati
        bergs = pygame.Surface((w, h), pygame.SRCALPHA)
        for _ in range(22):
            bx = rng.randint(0, w)
            bw, bh = rng.randint(70, 190), rng.randint(50, 150)
            base = int(h * 0.76)
            col = (rng.randint(168, 204), rng.randint(206, 232), rng.randint(228, 250))
            pts = [(bx - bw // 2, base)]
            for k in range(4):
                pts.append((bx - bw // 2 + k * bw // 3 + rng.randint(-12, 12),
                            base - rng.randint(bh // 2, bh)))
            pts.append((bx + bw // 2, base))
            pygame.draw.polygon(bergs, col, pts)
            pygame.draw.polygon(bergs, _shade(col, 0.82),
                                [(bx, base - bh // 2), (bx + bw // 2, base), (bx, base)])
        self.add(bergs, PARALLAX_MOUNTAIN_NEAR_SPEED)

        gh = SCREEN_HEIGHT - self.ground_y + 5
        ice = pygame.Surface((w, gh), pygame.SRCALPHA)
        ice.fill((222, 236, 248))
        for _ in range(150):  # crepe
            x, y = rng.randint(0, w), rng.randint(0, gh)
            pygame.draw.line(ice, (186, 208, 228), (x, y),
                             (x + rng.randint(-40, 40), y + rng.randint(-6, 6)), 1)
        self.add(ice, PARALLAX_GROUND_SPEED, self.ground_y)


class SpaceBiome(Biome):
    """Spazio profondo: nebulose, campi stellari e pianeti."""

    name = "spazio"
    # Nebulose e stelle coprono tutta l'altezza: il ritaglio guadagnerebbe
    # poco e la build dello spazio e' sincrona (non passa dal prebuild)
    crop_layers = False

    def build(self):
        rng = self.rng
        w, h = self.tile_w, SCREEN_HEIGHT
        self.sky = _vertical_gradient(SCREEN_WIDTH, h, (6, 4, 22), (18, 8, 34))

        # Nebulose: macchie di colore diffuso
        neb = pygame.Surface((w, h), pygame.SRCALPHA)
        for _ in range(7):
            nx, ny = rng.randint(0, w), rng.randint(0, h)
            col = rng.choice([(120, 40, 160), (40, 80, 190), (180, 50, 90), (30, 140, 150)])
            for _ in range(90):
                r = rng.randint(50, 210)
                pygame.draw.circle(neb, (*col, 14),
                                   (nx + rng.randint(-190, 190),
                                    ny + rng.randint(-130, 130)), r)
        self.add(neb, PARALLAX_MOUNTAIN_FAR_SPEED * 0.5)

        # Tre campi stellari a profondita' diverse
        for cnt, speed, mr in ((320, PARALLAX_MOUNTAIN_FAR_SPEED, 1),
                               (170, PARALLAX_MOUNTAIN_MID_SPEED, 2),
                               (80, PARALLAX_MOUNTAIN_NEAR_SPEED, 3)):
            field = pygame.Surface((w, h), pygame.SRCALPHA)
            _draw_star_field(field, w, h, cnt, rng, max_r=mr)
            self.add(field, speed)

        # Pianeti con terminatore e anelli
        planets = pygame.Surface((w, h), pygame.SRCALPHA)
        for _ in range(4):
            r = rng.randint(50, 130)
            # Margine di 2r dai bordi del tile: un pianeta tagliato a meta'
            # renderebbe visibile la giunzione dello scroll seamless
            px = rng.randint(int(r * 2), int(w - r * 2))
            py = rng.randint(60, h - 160)
            base = rng.choice([(180, 110, 70), (90, 130, 190), (150, 90, 150),
                               (110, 160, 120)])
            pygame.draw.circle(planets, base, (px, py), r)
            for band in range(-r, r, 9):  # bande atmosferiche
                bw = int(math.sqrt(max(0, r * r - band * band)))
                pygame.draw.line(planets, _shade(base, rng.uniform(0.85, 1.15)),
                                 (px - bw, py + band), (px + bw, py + band), 7)
            pygame.draw.circle(planets, (0, 0, 0, 130), (px + r // 3, py), r)
            pygame.draw.circle(planets, base, (px, py), r, 2)
            if rng.random() < 0.4:  # anelli
                for k in range(4):
                    pygame.draw.ellipse(planets, (*_shade(base, 1.4), 120),
                                        (px - r * 1.8, py - r // 4 + k * 4,
                                         r * 3.6, r // 2), 2)
        self.add(planets, PARALLAX_TREES_SPEED * 0.4)

        # Detriti / meteore vicine
        debris = pygame.Surface((w, h), pygame.SRCALPHA)
        for _ in range(45):
            dx, dy = rng.randint(0, w), rng.randint(0, h)
            dr = rng.randint(2, 7)
            col = (rng.randint(90, 150),) * 3
            pygame.draw.circle(debris, col, (dx, dy), dr)
            pygame.draw.circle(debris, _shade(col, 0.6), (dx + 1, dy + 1), dr - 1)
        self.add(debris, PARALLAX_GROUND_SPEED)


# Biomi terrestri, mescolati a ogni partita. Lo spazio e' riservato al finale.
SURFACE_BIOMES = [MountainBiome, SeaBiome, CityBiome,
                  JungleBiome, DesertBiome, ArcticBiome]

# Anticipo (ms) con cui il bioma successivo viene costruito in un thread:
# costruirlo nel frame dello switch (fino a ~90 ms) causava uno scatto
BIOME_PREBUILD_LEAD = 5000.0


# ── Gestore parallasse ──

class ParallaxBackground:
    """Sfondo parallasse multi-bioma con crossfade automatico.

    Ogni `BIOME_DURATION` ms passa al bioma successivo con una dissolvenza
    di `BIOME_FADE_DURATION` ms. Dal livello `BIOME_SPACE_LEVEL` in poi
    lo scenario diventa lo spazio profondo e non cambia piu'.
    """

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self._queue = self._new_queue()
        self.current = self._make(self._queue.pop(0))
        self.next: Biome | None = None
        self.timer = 0.0          # ms trascorsi nel bioma corrente
        self.fade = 0.0           # 0..1 avanzamento crossfade
        self.space_mode = False
        self._fade_surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT)).convert_alpha()
        self._prebuild = None     # (thread, risultato, classe, seed, stato da ripristinare)

    # -- costruzione biomi --

    def _new_queue(self) -> list:
        """Ordine casuale dei biomi terrestri, senza ripetere l'ultimo."""
        q = SURFACE_BIOMES[:]
        self.rng.shuffle(q)
        if getattr(self, "current", None) and type(self.current) is q[0] and len(q) > 1:
            q[0], q[1] = q[1], q[0]
        return q

    def _make(self, cls) -> Biome:
        return cls(random.Random(self.rng.randrange(1 << 30)))

    def _next_class(self):
        if not self._queue:
            self._queue = self._new_queue()
        return self._queue.pop(0)

    def _start_fade(self, cls):
        if self.next is None:
            self.next = self._make(cls)
            self.fade = 0.0

    def _start_prebuild(self):
        """Avvia in un thread la costruzione del prossimo bioma terrestre.

        Classe e seed si estraggono qui, nel main thread, nello stesso ordine
        di _make(): bioma e stato di self.rng sono identici a una costruzione
        sincrona allo switch. Le build usano solo il proprio random.Random e
        non convertono superfici, quindi possono girare fuori dal main thread.
        """
        undo = (self.rng.getstate(), list(self._queue))
        cls = self._next_class()
        seed = self.rng.randrange(1 << 30)
        result = {}

        def build():
            result['biome'] = cls(random.Random(seed))

        thread = threading.Thread(target=build, daemon=True)
        thread.start()
        self._prebuild = (thread, result, cls, seed, undo)

    def _finish_prebuild(self) -> Biome:
        """Attende il prebuild (di norma gia' concluso) e ne ritorna il bioma."""
        thread, result, cls, seed, _undo = self._prebuild
        self._prebuild = None
        thread.join()
        if 'biome' not in result:
            # Il thread e' fallito: ricostruisce qui, rialzando l'errore
            return cls(random.Random(seed))
        return result['biome']

    def _discard_prebuild(self):
        """Scarta il prebuild in corso e ripristina rng e coda come prima
        dell'estrazione, cosi' il seed dello spazio resta quello di sempre."""
        if self._prebuild is None:
            return
        rng_state, queue = self._prebuild[4]
        self.rng.setstate(rng_state)
        self._queue = queue
        self._prebuild = None

    # -- API pubblica --

    @property
    def biome_name(self) -> str:
        """Nome del bioma dominante (utile per HUD/debug)."""
        if self.next is not None and self.fade > 0.5:
            return self.next.name
        return self.current.name

    def update(self, dt: float, level: int = 1):
        """Avanza scroll e timer. `dt` e' normalizzato (~1.0 a FPS pieni)."""
        ms = dt * (1000.0 / 60.0)
        self.timer += ms

        # Transizione definitiva verso lo spazio agli ultimi livelli
        if not self.space_mode and level >= BIOME_SPACE_LEVEL:
            self.space_mode = True
            self.next = None
            self._discard_prebuild()
            self._start_fade(SpaceBiome)
            self.timer = 0.0

        if self.next is None and not self.space_mode:
            # Il bioma successivo si costruisce in background prima dello switch
            if self._prebuild is None and self.timer >= BIOME_DURATION - BIOME_PREBUILD_LEAD:
                self._start_prebuild()
            if self.timer >= BIOME_DURATION:
                self.next = self._finish_prebuild()
                self.fade = 0.0

        self.current.update(dt)
        if self.next is not None:
            self.next.update(dt)
            self.fade += ms / max(1.0, BIOME_FADE_DURATION)
            if self.fade >= 1.0:
                self.current = self.next
                self.next = None
                self.fade = 0.0
                self.timer = 0.0

    def draw(self, surface: pygame.Surface):
        self.current.draw(surface)
        if self.next is not None:
            self._fade_surf.fill((0, 0, 0, 0))
            self.next.draw(self._fade_surf)
            self._fade_surf.set_alpha(int(255 * min(1.0, self.fade)))
            surface.blit(self._fade_surf, (0, 0))
