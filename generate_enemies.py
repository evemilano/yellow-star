"""Genera 16 sprite nemici pixel-art e li salva in assets/images/enemies/.
Ogni nave viene disegnata su una griglia logica poi scalata 2x per effetto pixel-art.
Le navi sono orientate verso sinistra (puntano a sx, arrivano da destra).
"""
import os
import pygame

pygame.init()
pygame.display.set_mode((1, 1))  # serve per le surface

OUT_DIR = os.path.join(os.path.dirname(__file__), "assets", "images", "enemies")
os.makedirs(OUT_DIR, exist_ok=True)

PX = 3  # dimensione pixel logico
GW, GH = 20, 28  # griglia logica (larghezza, altezza) - nave punta in alto
# Dopo rotazione 90° CW: la nave punterà a sinistra, dimensione finale ~ 28*PX x 20*PX = 84x60


def px(surf, x, y, color):
    if 0 <= x < GW and 0 <= y < GH:
        pygame.draw.rect(surf, color, (x * PX, y * PX, PX, PX))


def hline(surf, x1, x2, y, color):
    for x in range(x1, x2 + 1):
        px(surf, x, y, color)


def vline(surf, x, y1, y2, color):
    for y in range(y1, y2 + 1):
        px(surf, x, y, color)


def fill(surf, x1, y1, x2, y2, color):
    for y in range(y1, y2 + 1):
        hline(surf, x1, x2, y, color)


def make_surf():
    return pygame.Surface((GW * PX, GH * PX), pygame.SRCALPHA)


def rotate_to_left(surf):
    """Ruota la nave (che punta in alto) di 90° orario -> punta a sinistra."""
    return pygame.transform.rotate(surf, 90)


# ═══════════════════════════════════════════════════════
# 16 MODELLI DI NAVI NEMICHE
# ═══════════════════════════════════════════════════════

def enemy_01():
    """Rossa con booster laterali - stile razzo."""
    s = make_surf()
    cx = 10
    # Booster sinistro
    fill(s, cx - 6, 18, cx - 5, 26, (180, 40, 20))
    fill(s, cx - 6, 26, cx - 5, 27, (255, 140, 40))
    px(s, cx - 6, 16, (140, 30, 15))
    px(s, cx - 5, 16, (140, 30, 15))
    px(s, cx - 6, 17, (160, 35, 18))
    px(s, cx - 5, 17, (160, 35, 18))
    # Booster destro
    fill(s, cx + 5, 18, cx + 6, 26, (180, 40, 20))
    fill(s, cx + 5, 26, cx + 6, 27, (255, 140, 40))
    px(s, cx + 5, 16, (140, 30, 15))
    px(s, cx + 6, 16, (140, 30, 15))
    px(s, cx + 5, 17, (160, 35, 18))
    px(s, cx + 6, 17, (160, 35, 18))
    # Fusoliera centrale
    fill(s, cx - 2, 4, cx + 2, 24, (160, 50, 30))
    fill(s, cx - 1, 2, cx + 1, 24, (190, 60, 35))
    # Muso
    px(s, cx, 1, (200, 70, 40))
    px(s, cx, 0, (220, 80, 50))
    # Cabina
    fill(s, cx - 1, 5, cx + 1, 7, (200, 50, 30))
    px(s, cx, 5, (255, 80, 60))
    # Fiamma principale
    fill(s, cx - 1, 25, cx + 1, 27, (255, 160, 40))
    px(s, cx, 27, (255, 200, 80))
    return rotate_to_left(s)


def enemy_02():
    """Caccia a delta - marrone/beige con ali swept."""
    s = make_surf()
    cx = 10
    # Fusoliera
    fill(s, cx - 1, 3, cx + 1, 22, (150, 130, 100))
    fill(s, cx - 2, 8, cx + 2, 20, (130, 110, 85))
    px(s, cx, 2, (170, 150, 120))
    px(s, cx, 1, (180, 160, 130))
    # Ali swept
    for i in range(7):
        px(s, cx - 3 - i, 12 + i, (130, 110, 85))
        px(s, cx + 3 + i, 12 + i, (130, 110, 85))
        if i < 6:
            px(s, cx - 3 - i, 13 + i, (110, 95, 75))
            px(s, cx + 3 + i, 13 + i, (110, 95, 75))
    # Dettagli arancioni
    px(s, cx - 2, 10, (200, 80, 30))
    px(s, cx + 2, 10, (200, 80, 30))
    fill(s, cx - 1, 21, cx + 1, 23, (200, 80, 30))
    # Cabina
    px(s, cx, 5, (100, 80, 60))
    px(s, cx, 6, (120, 100, 70))
    return rotate_to_left(s)


def enemy_03():
    """Nave blu scuro con cockpit turchese grande."""
    s = make_surf()
    cx = 10
    HULL = (40, 55, 80)
    HULL_L = (60, 75, 105)
    COCK = (80, 200, 180)
    COCK_H = (140, 240, 220)
    # Fusoliera
    fill(s, cx - 2, 4, cx + 2, 22, HULL)
    fill(s, cx - 1, 3, cx + 1, 22, HULL_L)
    px(s, cx, 2, HULL_L)
    # Cockpit grande
    fill(s, cx - 1, 5, cx + 1, 10, COCK)
    px(s, cx, 5, COCK_H)
    px(s, cx, 6, COCK_H)
    # Ali
    fill(s, cx - 5, 12, cx - 3, 18, HULL)
    fill(s, cx + 3, 12, cx + 5, 18, HULL)
    px(s, cx - 4, 11, HULL_L)
    px(s, cx + 4, 11, HULL_L)
    # Propulsori
    fill(s, cx - 4, 19, cx - 3, 21, (50, 50, 60))
    fill(s, cx + 3, 19, cx + 4, 21, (50, 50, 60))
    px(s, cx - 4, 22, (255, 140, 40))
    px(s, cx - 3, 22, (255, 140, 40))
    px(s, cx + 3, 22, (255, 140, 40))
    px(s, cx + 4, 22, (255, 140, 40))
    # Fiamma centrale
    fill(s, cx - 1, 23, cx + 1, 24, (255, 180, 60))
    # Antenna
    px(s, cx, 1, (180, 50, 50))
    px(s, cx, 0, (220, 60, 60))
    return rotate_to_left(s)


def enemy_04():
    """Razzo snello blu/turchese con cockpit luminoso."""
    s = make_surf()
    cx = 10
    HULL = (35, 50, 75)
    HULL_L = (55, 70, 100)
    COCK = (100, 220, 200)
    # Fusoliera snella
    fill(s, cx - 1, 3, cx + 1, 23, HULL)
    vline(s, cx, 1, 23, HULL_L)
    px(s, cx, 0, HULL_L)
    # Cockpit
    fill(s, cx - 1, 4, cx + 1, 8, COCK)
    px(s, cx, 4, (160, 255, 240))
    # Alette piccole
    fill(s, cx - 4, 16, cx - 2, 19, HULL)
    fill(s, cx + 2, 16, cx + 4, 19, HULL)
    # Punte rosse
    px(s, cx, 0, (220, 50, 40))
    px(s, cx - 4, 16, (220, 50, 40))
    px(s, cx + 4, 16, (220, 50, 40))
    # Propulsore
    fill(s, cx - 1, 24, cx + 1, 25, (255, 160, 50))
    px(s, cx, 26, (255, 200, 80))
    return rotate_to_left(s)


def enemy_05():
    """Nave cargo pesante - marrone/arancione con ali larghe."""
    s = make_surf()
    cx = 10
    HULL = (120, 90, 55)
    HULL_L = (150, 115, 70)
    ORANGE = (200, 120, 40)
    # Fusoliera larga
    fill(s, cx - 3, 5, cx + 3, 22, HULL)
    fill(s, cx - 2, 4, cx + 2, 22, HULL_L)
    px(s, cx, 3, HULL_L)
    # Ali
    for i in range(5):
        fill(s, cx - 5 - i, 10 + i, cx - 4, 12 + i, HULL)
        fill(s, cx + 4, 10 + i, cx + 5 + i, 12 + i, HULL)
    # Dettagli arancioni sulle ali
    px(s, cx - 6, 12, ORANGE)
    px(s, cx + 6, 12, ORANGE)
    px(s, cx - 7, 13, ORANGE)
    px(s, cx + 7, 13, ORANGE)
    # Cabina
    fill(s, cx - 1, 6, cx + 1, 8, (80, 60, 40))
    px(s, cx, 6, (200, 60, 40))
    # Propulsori
    fill(s, cx - 2, 23, cx + 2, 25, (255, 160, 40))
    px(s, cx - 1, 26, (255, 200, 80))
    px(s, cx + 1, 26, (255, 200, 80))
    return rotate_to_left(s)


def enemy_06():
    """Caccia grigio/marrone con pattern mimetico."""
    s = make_surf()
    cx = 10
    C1 = (100, 95, 80)
    C2 = (130, 120, 95)
    C3 = (80, 75, 60)
    # Fusoliera
    fill(s, cx - 2, 4, cx + 2, 21, C1)
    fill(s, cx - 1, 3, cx + 1, 21, C2)
    # Pattern mimetico
    px(s, cx - 2, 8, C3)
    px(s, cx + 1, 10, C3)
    px(s, cx - 1, 14, C3)
    px(s, cx + 2, 16, C3)
    px(s, cx, 2, C2)
    # Ali swept
    for i in range(6):
        px(s, cx - 3 - i, 11 + i, C1)
        px(s, cx + 3 + i, 11 + i, C1)
        px(s, cx - 3 - i, 12 + i, C3)
        px(s, cx + 3 + i, 12 + i, C3)
    # Dettagli rossi
    fill(s, cx - 1, 22, cx + 1, 23, (200, 80, 30))
    px(s, cx, 1, (200, 80, 30))
    return rotate_to_left(s)


def enemy_07():
    """Nave verde/turchese compatta con cockpit quadrato."""
    s = make_surf()
    cx = 10
    HULL = (50, 100, 90)
    HULL_L = (70, 130, 115)
    COCK = (150, 230, 200)
    # Fusoliera compatta
    fill(s, cx - 3, 8, cx + 3, 20, HULL)
    fill(s, cx - 2, 6, cx + 2, 20, HULL_L)
    fill(s, cx - 1, 5, cx + 1, 6, HULL_L)
    px(s, cx, 4, HULL_L)
    # Cockpit quadrato grande
    fill(s, cx - 2, 8, cx + 2, 12, COCK)
    fill(s, cx - 1, 8, cx + 1, 11, (180, 250, 230))
    # Ali tozze
    fill(s, cx - 6, 12, cx - 4, 17, HULL)
    fill(s, cx + 4, 12, cx + 6, 17, HULL)
    # Propulsori
    fill(s, cx - 2, 21, cx + 2, 23, (50, 50, 60))
    hline(s, cx - 2, cx + 2, 24, (255, 160, 40))
    return rotate_to_left(s)


def enemy_08():
    """Nave viola scuro con armi laterali - stile boss."""
    s = make_surf()
    cx = 10
    HULL = (50, 35, 70)
    HULL_L = (75, 55, 100)
    # Fusoliera
    fill(s, cx - 2, 4, cx + 2, 22, HULL)
    fill(s, cx - 1, 3, cx + 1, 22, HULL_L)
    px(s, cx, 2, HULL_L)
    # Ali con cannoni
    fill(s, cx - 5, 10, cx - 3, 16, HULL)
    fill(s, cx + 3, 10, cx + 5, 16, HULL)
    # Cannoni
    vline(s, cx - 5, 7, 10, (100, 100, 110))
    vline(s, cx + 5, 7, 10, (100, 100, 110))
    # Cockpit tondo
    px(s, cx, 5, (180, 255, 180))
    px(s, cx - 1, 6, (180, 255, 180))
    px(s, cx + 1, 6, (180, 255, 180))
    px(s, cx, 6, (220, 255, 220))
    # Teschio (dettaglio minaccioso)
    px(s, cx - 1, 8, (220, 220, 220))
    px(s, cx + 1, 8, (220, 220, 220))
    px(s, cx, 9, (220, 220, 220))
    # Propulsore
    fill(s, cx - 1, 23, cx + 1, 24, (255, 140, 40))
    px(s, cx, 25, (255, 200, 80))
    # Antenna rossa
    px(s, cx, 1, (200, 50, 50))
    return rotate_to_left(s)


def enemy_09():
    """Nave blu chiaro aerodinamica lunga."""
    s = make_surf()
    cx = 10
    HULL = (60, 110, 130)
    HULL_L = (90, 145, 165)
    COCK = (160, 220, 240)
    # Fusoliera lunga
    fill(s, cx - 1, 2, cx + 1, 24, HULL)
    vline(s, cx, 1, 24, HULL_L)
    px(s, cx, 0, HULL_L)
    # Sezione centrale più larga
    fill(s, cx - 2, 10, cx + 2, 18, HULL)
    # Ali
    for i in range(5):
        px(s, cx - 3 - i, 12 + i, HULL)
        px(s, cx + 3 + i, 12 + i, HULL)
    # Cockpit
    fill(s, cx - 1, 4, cx + 1, 7, COCK)
    px(s, cx, 3, (200, 240, 255))
    # Propulsore
    px(s, cx, 25, (255, 160, 40))
    px(s, cx - 1, 25, (200, 100, 30))
    px(s, cx + 1, 25, (200, 100, 30))
    # Dettaglio blu
    px(s, cx, 0, (50, 80, 200))
    return rotate_to_left(s)


def enemy_10():
    """Caccia turchese compatto con ali a freccia."""
    s = make_surf()
    cx = 10
    HULL = (40, 90, 85)
    HULL_L = (60, 120, 110)
    # Fusoliera
    fill(s, cx - 1, 5, cx + 1, 20, HULL)
    vline(s, cx, 3, 20, HULL_L)
    px(s, cx, 2, HULL_L)
    # Ali a freccia
    fill(s, cx - 4, 12, cx - 2, 14, HULL)
    fill(s, cx + 2, 12, cx + 4, 14, HULL)
    fill(s, cx - 6, 14, cx - 3, 16, HULL)
    fill(s, cx + 3, 14, cx + 6, 16, HULL)
    # Punte alari
    px(s, cx - 6, 13, HULL_L)
    px(s, cx + 6, 13, HULL_L)
    # Cockpit
    px(s, cx, 6, (140, 230, 210))
    px(s, cx, 7, (100, 200, 180))
    # Propulsore
    fill(s, cx - 1, 21, cx + 1, 22, (255, 160, 40))
    # Antenna turchese
    px(s, cx, 1, (80, 200, 200))
    return rotate_to_left(s)


def enemy_11():
    """Nave viola/arancione triangolare con ali grandi."""
    s = make_surf()
    cx = 10
    HULL = (55, 40, 80)
    HULL_L = (80, 60, 110)
    ORANGE = (220, 140, 40)
    # Fusoliera
    fill(s, cx - 1, 4, cx + 1, 20, HULL)
    vline(s, cx, 3, 20, HULL_L)
    # Ali grandi triangolari
    for i in range(8):
        px(s, cx - 2 - i, 10 + i, HULL)
        px(s, cx + 2 + i, 10 + i, HULL)
        if i > 0:
            px(s, cx - 1 - i, 10 + i, HULL_L)
            px(s, cx + 1 + i, 10 + i, HULL_L)
    # Accenti arancioni
    px(s, cx, 3, ORANGE)
    px(s, cx - 2, 18, ORANGE)
    px(s, cx + 2, 18, ORANGE)
    # Cockpit
    px(s, cx, 6, (200, 180, 100))
    px(s, cx, 7, (180, 160, 80))
    # Propulsore
    fill(s, cx - 1, 21, cx + 1, 22, (255, 140, 40))
    px(s, cx, 23, (255, 200, 80))
    return rotate_to_left(s)


def enemy_12():
    """Nave pesante arancione/marrone con forma a razzo."""
    s = make_surf()
    cx = 10
    HULL = (170, 120, 50)
    HULL_L = (200, 150, 70)
    DARK = (120, 80, 35)
    # Fusoliera larga
    fill(s, cx - 3, 5, cx + 3, 22, HULL)
    fill(s, cx - 2, 3, cx + 2, 22, HULL_L)
    fill(s, cx - 1, 2, cx + 1, 3, HULL_L)
    px(s, cx, 1, HULL_L)
    # Bande scure
    hline(s, cx - 3, cx + 3, 10, DARK)
    hline(s, cx - 3, cx + 3, 15, DARK)
    # Cockpit
    fill(s, cx - 1, 4, cx + 1, 6, (100, 70, 30))
    px(s, cx, 4, (200, 100, 40))
    # Ali piccole
    fill(s, cx - 5, 14, cx - 4, 18, HULL)
    fill(s, cx + 4, 14, cx + 5, 18, HULL)
    # Propulsori laterali
    px(s, cx - 4, 19, (255, 140, 40))
    px(s, cx + 4, 19, (255, 140, 40))
    # Fiamma centrale
    fill(s, cx - 2, 23, cx + 2, 25, (255, 160, 40))
    hline(s, cx - 1, cx + 1, 26, (255, 200, 80))
    return rotate_to_left(s)


def enemy_13():
    """Nave marrone/grigia con grandi ali swept - cacciabombardiere."""
    s = make_surf()
    cx = 10
    HULL = (100, 85, 70)
    HULL_L = (130, 115, 95)
    # Fusoliera
    fill(s, cx - 2, 4, cx + 2, 22, HULL)
    fill(s, cx - 1, 3, cx + 1, 22, HULL_L)
    px(s, cx, 2, HULL_L)
    # Ali sweep grandi
    for i in range(7):
        fill(s, cx - 3 - i, 11 + i, cx - 3, 12 + i, HULL)
        fill(s, cx + 3, 11 + i, cx + 3 + i, 12 + i, HULL)
    # Dettagli arancioni punta ali
    px(s, cx - 9, 17, (220, 120, 30))
    px(s, cx + 9, 17, (220, 120, 30))
    # Cabina
    fill(s, cx - 1, 5, cx + 1, 7, (70, 60, 50))
    px(s, cx, 5, (200, 80, 40))
    # Propulsore
    fill(s, cx - 1, 23, cx + 1, 25, (255, 160, 40))
    # Antenna
    px(s, cx, 1, (80, 200, 180))
    return rotate_to_left(s)


def enemy_14():
    """Caccia blu scuro snello con doppia coda."""
    s = make_surf()
    cx = 10
    HULL = (40, 50, 80)
    HULL_L = (60, 75, 110)
    # Fusoliera
    fill(s, cx - 1, 3, cx + 1, 22, HULL)
    vline(s, cx, 2, 22, HULL_L)
    px(s, cx, 1, HULL_L)
    # Ali
    fill(s, cx - 5, 11, cx - 2, 14, HULL)
    fill(s, cx + 2, 11, cx + 5, 14, HULL)
    # Doppia coda
    fill(s, cx - 4, 18, cx - 3, 22, HULL)
    fill(s, cx + 3, 18, cx + 4, 22, HULL)
    px(s, cx - 4, 23, (255, 140, 40))
    px(s, cx + 4, 23, (255, 140, 40))
    # Cockpit
    px(s, cx, 4, (200, 140, 40))
    px(s, cx, 5, (180, 120, 30))
    # Dettaglio rosso
    px(s, cx, 1, (200, 50, 40))
    # Propulsore
    fill(s, cx - 1, 23, cx + 1, 24, (255, 160, 40))
    return rotate_to_left(s)


def enemy_15():
    """Nave rossa/arancione con ali delta e propulsore grosso."""
    s = make_surf()
    cx = 10
    HULL = (150, 60, 30)
    HULL_L = (180, 80, 40)
    ORANGE = (220, 140, 50)
    # Fusoliera
    fill(s, cx - 2, 4, cx + 2, 22, HULL)
    fill(s, cx - 1, 3, cx + 1, 22, HULL_L)
    px(s, cx, 2, HULL_L)
    # Ali delta
    for i in range(6):
        fill(s, cx - 3 - i, 12 + i, cx - 3, 13 + i, HULL)
        fill(s, cx + 3, 12 + i, cx + 3 + i, 13 + i, HULL)
    # Accenti arancioni
    px(s, cx - 2, 6, ORANGE)
    px(s, cx + 2, 6, ORANGE)
    hline(s, cx - 1, cx + 1, 10, ORANGE)
    # Cockpit
    px(s, cx, 4, (200, 220, 180))
    px(s, cx, 5, (180, 200, 160))
    # Propulsore grosso
    fill(s, cx - 2, 23, cx + 2, 26, (200, 80, 30))
    fill(s, cx - 1, 24, cx + 1, 27, (255, 160, 50))
    px(s, cx, 27, (255, 220, 100))
    return rotate_to_left(s)


def enemy_16():
    """Nave dorata/bronzo pesante con scudo frontale."""
    s = make_surf()
    cx = 10
    HULL = (160, 130, 60)
    HULL_L = (190, 160, 80)
    DARK = (100, 80, 40)
    # Fusoliera larga
    fill(s, cx - 3, 6, cx + 3, 22, HULL)
    fill(s, cx - 2, 4, cx + 2, 22, HULL_L)
    # Scudo frontale (largo)
    fill(s, cx - 4, 4, cx + 4, 6, DARK)
    fill(s, cx - 3, 3, cx + 3, 5, HULL)
    px(s, cx, 2, HULL_L)
    px(s, cx, 1, HULL_L)
    # Ali corte e larghe
    fill(s, cx - 6, 12, cx - 4, 16, HULL)
    fill(s, cx + 4, 12, cx + 6, 16, HULL)
    # Dettagli
    px(s, cx - 5, 12, (220, 120, 30))
    px(s, cx + 5, 12, (220, 120, 30))
    # Cockpit
    fill(s, cx - 1, 6, cx + 1, 8, DARK)
    px(s, cx, 6, (200, 160, 60))
    # Propulsori doppi
    fill(s, cx - 2, 23, cx - 1, 25, (255, 140, 40))
    fill(s, cx + 1, 23, cx + 2, 25, (255, 140, 40))
    px(s, cx - 2, 26, (255, 200, 80))
    px(s, cx + 2, 26, (255, 200, 80))
    return rotate_to_left(s)


# ── Genera e salva tutti ──
ENEMIES = [
    enemy_01, enemy_02, enemy_03, enemy_04,
    enemy_05, enemy_06, enemy_07, enemy_08,
    enemy_09, enemy_10, enemy_11, enemy_12,
    enemy_13, enemy_14, enemy_15, enemy_16,
]

for i, fn in enumerate(ENEMIES, 1):
    surf = fn()
    path = os.path.join(OUT_DIR, f"enemy_{i:02d}.png")
    pygame.image.save(surf, path)
    print(f"Salvato: {path}  ({surf.get_width()}x{surf.get_height()})")

print(f"\nGenerati {len(ENEMIES)} sprite nemici!")
pygame.quit()
