# ── Yellow Star - Configurazione ──

TITLE = "Yellow Star"

# Risoluzione e display
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 900
FPS = 30

# Colori
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 50, 50)
CYAN = (0, 255, 255)
GREEN = (0, 200, 80)
DARK_GREEN = (20, 80, 30)
BROWN = (100, 60, 30)
SKY_BLUE = (100, 160, 220)
LIGHT_BLUE = (150, 200, 240)
MOUNTAIN_FAR = (80, 100, 130)
MOUNTAIN_MID = (60, 80, 100)
MOUNTAIN_NEAR = (40, 60, 80)
GROUND_COLOR = (60, 130, 50)

# Giocatore
PLAYER_SPEED = 6
PLAYER_SIZE = (96, 40)
PLAYER_COLOR = CYAN
PLAYER_SHOOT_DELAY = 250  # millisecondi tra un colpo e l'altro
PLAYER_LIVES = 3
PLAYER_INVINCIBLE_TIME = 2000  # ms di invincibilità dopo essere colpito

# Proiettili
BULLET_SPEED = 12
BULLET_SIZE = (14, 4)
BULLET_COLOR = YELLOW

# Nemici (astronavi) - 16 modelli diversi in assets/images/enemies/
ENEMY_SPEED_MIN = 3
ENEMY_SPEED_MAX = 6
ENEMY_MODELS = 16
ENEMY_SPAWN_DELAY = 1200  # millisecondi tra uno spawn e l'altro

# Asteroidi (ostacoli da schivare)
ASTEROID_SPEED_MIN = 2
ASTEROID_SPEED_MAX = 4
ASTEROID_SIZE_MIN = 20
ASTEROID_SIZE_MAX = 45
ASTEROID_COLOR = (140, 120, 100)
ASTEROID_SPAWN_DELAY = 2500  # millisecondi

# Missili bonus (power-up)
MISSILE_SPEED = 14
MISSILE_KILL_COUNT = 5          # numero di nemici distrutti per missile
MISSILE_POWERUP_SPAWN_DELAY = 8000  # ms tra uno spawn e l'altro
MISSILE_POWERUP_SPEED = 2      # velocità scorrimento del power-up

# ── Livelli ──
LEVEL_SCORE_BASE = 500         # punti per passare dal livello 1 al 2
LEVEL_SCORE_GROWTH = 1.4       # moltiplicatore per ogni livello successivo
LEVEL_DIFFICULTY_FACTOR = 1.12  # fattore di scala difficoltà per livello
LEVEL_MAX = 100                  # livello massimo raggiungibile

# ── Potenziamenti (drop dai nemici) ──
POWERUP_DROP_CHANCE = 0.15      # 15% probabilità di drop
POWERUP_SPEED = 2               # velocità scorrimento del power-up
BOMB_FREEZE_DURATION = 1000     # ms di freeze spawn dopo bomba (lv8+)
BOMB_FIELD_DURATION = 2000      # ms di campo protettivo dopo bomba (lv10)

# ── Proiettili nemici ──
ENEMY_BULLET_SPEED = 7             # pixel/tick (si muove verso sinistra)
ENEMY_SHOOT_DELAY_BASIC = 2200     # ms tra un colpo e l'altro
ENEMY_SHOOT_DELAY_AIMED = 2800     # aimed: meno frequente, mira al player
ENEMY_SHOOT_DELAY_BURST = 3200     # burst: 3 proiettili a ventaglio
ENEMY_SHOOT_DELAY_FAST = 1400      # fast: molto frequente
ENEMY_BURST_ANGLES = (-12, 0, 12)  # gradi offset per burst

# Probabilità che un nemico sia shooter (indice = livello - 1)
ENEMY_SHOOTER_CHANCE = [
    0.00, 0.00, 0.05, 0.08, 0.10,  # lv 1-5
    0.13, 0.16, 0.20, 0.24, 0.28,  # lv 6-10
    0.32, 0.35, 0.38, 0.42, 0.45,  # lv 11-15
    0.48, 0.52, 0.55, 0.58, 0.62,  # lv 16-20
]

# Livello di sblocco per ogni tipo di shooter
ENEMY_SHOOTER_UNLOCK = {"basic": 3, "fast": 4, "aimed": 6, "burst": 10}

# Parallasse - paesaggio
PARALLAX_MOUNTAIN_FAR_SPEED = 0.3
PARALLAX_MOUNTAIN_MID_SPEED = 0.6
PARALLAX_MOUNTAIN_NEAR_SPEED = 1.0
PARALLAX_TREES_SPEED = 1.8
PARALLAX_GROUND_SPEED = 2.5
