# Yellow Star - AI Edition

A side-scrolling space shooter built with **Pygame-CE** featuring a full **NEAT neuroevolution** pipeline that evolves neural networks to play the game autonomously. Includes human demonstration recording, imitation learning, cooperative human+AI gameplay, and a real-time training dashboard.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Game Modes](#game-modes)
- [Gameplay Mechanics](#gameplay-mechanics)
  - [Player Controls](#player-controls)
  - [Enemies](#enemies)
  - [Obstacles](#obstacles)
  - [Power-ups & Upgrades](#power-ups--upgrades)
  - [Weapons](#weapons)
  - [Level Progression](#level-progression)
- [AI / Machine Learning System](#ai--machine-learning-system)
  - [Architecture Overview](#architecture-overview)
  - [Neural Network I/O](#neural-network-io)
  - [Fitness Function](#fitness-function)
  - [Imitation Learning](#imitation-learning)
  - [Training Pipeline](#training-pipeline)
  - [Anti-Stagnation Mechanisms](#anti-stagnation-mechanisms)
  - [Checkpointing & Persistence](#checkpointing--persistence)
- [Project Structure](#project-structure)
- [Configuration Reference](#configuration-reference)
- [Technical Details](#technical-details)
  - [Procedural Audio Synthesis](#procedural-audio-synthesis)
  - [Parallax Rendering](#parallax-rendering)
  - [Procedural Sprite Generation](#procedural-sprite-generation)
- [Requirements](#requirements)
- [License](#license)

---

## Features

- **4 Game Modes**: Human play, headless AI training, visual AI training, cooperative human+AI
- **NEAT Neuroevolution**: Evolving neural network topologies with speciation and crossover
- **Imitation Learning**: Human gameplay sessions are recorded and used to guide evolution
- **Multiprocess Training**: Parallel genome evaluation on `cpu_count - 1` worker processes (headless mode)
- **Real-Time Dashboard**: Live fitness metrics, fitness component breakdown, species breakdown, and behavior statistics during training
- **Procedural Audio**: All sound effects and background music generated at runtime via waveform synthesis (zero external audio files)
- **Procedural Sprites**: Enemy ships (pre-generated to PNG by a script), explosions, asteroids, projectiles and power-ups are generated programmatically
- **Biome Background**: Procedural parallax landscapes (6 surface biomes + a final space biome) with automatic crossfades
- **Upgrade System**: 4 upgrade categories (Engine, Weapon, Shield, Bomb) with 10 levels each
- **16 Enemy Models + 4 Shooting Behaviors**: Unique pixel-art ships; shooting behaviors unlock with the level and are assigned independently of the model
- **Dynamic Difficulty**: Exponential scaling across 100 levels

---

## Installation

```bash
# Clone the repository
git clone https://github.com/evemilano/yellow-star.git
cd yellow-star

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| `pygame-ce` | >= 2.5.0 | Game engine (Community Edition fork of Pygame) |
| `neat-python` | unpinned (latest) | NEAT neuroevolution library |
| Python | >= 3.10 | Runtime |

---

## Quick Start

```bash
# Run the game (activates venv automatically on Windows)
run.bat

# Or manually
python main.py
```

The main menu presents four options navigable with arrow keys (menu labels are in Italian: *Gioca*, *Allena AI*, *Vedi AI che si allena*, *Gioca con AI*):

| Key | Action |
|-----|--------|
| `UP` / `DOWN` | Navigate menu (wraps around) |
| `ENTER` | Select option |
| `ESC` / close window | Quit |

---

## Game Modes

### 1. Play (Human)
Standard keyboard-controlled gameplay. Every game is **recorded** for imitation learning and saved to `ai/human_sessions/` when it ends with a game over (leaving with `ESC` before game over discards the recording). Up to 20 sessions are retained (oldest deleted first).

### 2. Train AI (Headless)
Background NEAT training with no game rendering. Uses **multiprocessing** to evaluate genomes in parallel on `cpu_count - 1` worker processes; the population size is forced to **300**. A real-time dashboard is drawn in the game window (progress, fitness, fitness breakdown, species, AI behavior, efficiency); neat-python's `StdOutReporter` also prints per-generation statistics to the console. `ESC` or closing the window interrupts training and returns to the menu.

### 3. View AI Training (Visual)
Same NEAT training, but genomes are evaluated serially in the main process with live rendering of each genome playing (biome background, minimal HUD, training overlay). Only 1 frame out of 4 is drawn (`TRAIN_RENDER_SKIP = 4`) and there is no frame-rate cap. The population size comes from the NEAT config (150 in `ai/neat_config.txt`, or the value stored in the checkpoint being resumed).

### 4. Coop AI
Human and a trained AI agent play **simultaneously** on the same screen. Requires a saved `ai/best_genome.pkl` from prior training (otherwise the menu entry is greyed out and `ENTER` on it does nothing). Both ships share the game world and the score; each ship has its own lives, upgrades, missiles and bombs, and power-ups go to whichever ship picks them up. The AI ship is tinted blue with an "AI" label; aimed enemies always target the human ship. When the AI loses its 3 lives it is removed ("AI eliminata!"); the game ends only when the human loses. The human's actions are recorded for imitation learning as in Play mode.

---

## Gameplay Mechanics

### Player Controls

| Key | Action |
|-----|--------|
| `Arrow Keys` | Move (4 directions) |
| `SPACE` (hold) | Shoot |
| `M` | Fire missile (one per key press) |
| `B` | Detonate bomb |
| `ESC` | Return to main menu |

After game over: type a name (max 10 characters, `BACKSPACE` to delete), `ENTER` to save it (non-empty name) to the Top 10 leaderboard, then `ENTER` again to restart.

- **Speed**: Base 6 px/frame (at 60 FPS), +10% per engine level up to 2x (12 px/frame)
- **Health**: 3 lives + shield HP (0-5) from shield upgrades; the shield absorbs hits before lives
- **Invincibility**: 2.0–4.5 seconds after any hit (shield or life): 2 s base, +0.5 s every 2 shield levels

### Enemies

16 unique pixel-art enemy models (chosen at random) spawn from the right edge and scroll left at 3-6 px/frame (level 1). Four shooting behaviors unlock progressively:

| Type | Unlock Level | Base Fire Delay | Behavior |
|------|-------------|------------|----------|
| **Basic** | 3 | 2200 ms | Straight shot leftward |
| **Fast** | 4 | 1400 ms | Straight shot, high frequency |
| **Aimed** | 6 | 2800 ms | Calculates angle to the (human) player |
| **Burst** | 10 | 3200 ms | 3-bullet spread (-12, 0, +12 degrees) |

Fire delays are divided by the difficulty factor (`1.12 ^ (level - 1)`) with a floor of 600 ms. Enemies only fire once fully on screen, and the first shot is staggered by an extra 0.5-1.5 s.

Shooter probability per enemy is a lookup table indexed by level (`ENEMY_SHOOTER_CHANCE`): 0% at levels 1-2, 5% at level 3, rising step by step to 62% at level 20 and staying there for higher levels. A shooter's type is picked uniformly among the unlocked types. Ramming an enemy destroys it (no points) and damages the player.

### Obstacles

**Asteroids** are procedurally generated irregular polygons (7-10 vertices). Radius ranges from 20-45 px at level 1, growing by 2 px per level up to 40-65 px (reached at level 11). Speed starts at 2-4 px/frame and grows by +0.3 / +0.4 px/frame per level. Asteroids are indestructible except by bombs (bomb level 2+): bullets are absorbed with a spark, and colliding with an asteroid damages the player without destroying it.

### Power-ups & Upgrades

Enemies destroyed by bullets or missiles have a **15% drop chance** for one of 4 power-up types (chosen uniformly). Kills by bomb or by ramming never drop power-ups.

| Type | Color | Icon | Effect |
|------|-------|------|--------|
| **Engine** | Green | M | +10% speed per level (max 2x) |
| **Weapon** | Orange | W | +1 bullet on even levels (max 5), -15% fire delay on odd levels; changes the player ship sprite |
| **Shield** | Blue | S | +1 max shield HP every 2 levels (max 5), refills the shield, +invincibility duration |
| **Bomb** | Purple | B | +1 bomb (capacity +1 every 2 levels, max 5), unlocks effects |

Each category has **10 upgrade levels**. Key milestones:

**Weapon progression:**
| Level | Bullets | Fire Delay |
|-------|---------|-----------|
| 0 / 1 | 1 | 250 ms / 212 ms (-15%) |
| 2 / 3 | 2 | 250 ms / 212 ms |
| 4 / 5 | 3 | 250 ms / 212 ms |
| 6 / 7 | 4 | 250 ms / 212 ms |
| 8 / 9 | 5 | 250 ms / 212 ms |
| 10 | 5 | 175 ms (-30%) |

**Bomb unlock chain:**
| Level | Effect |
|-------|--------|
| 1+ | Destroys all enemies (100 pts each) and all enemy bullets |
| 2+ | Also destroys asteroids |
| 6+ | Double score on bomb kills (200 pts) |
| 8+ | Freezes enemy and asteroid spawns for 1 second |
| 10 | Protective field for 2 seconds (all player collisions ignored) |

### Weapons

- **Bullets**: 12 px/frame, animated 3-frame sprites with flame trail, 1-5 per volley depending on weapon level. Score: 100 pts/kill
- **Missiles**: Fly straight to the right at 14 px/frame and pierce through up to 5 enemies they touch (no homing). Score: 150 pts/kill. Ammo comes from floating missile pickups that spawn from the right edge every 8 s at level 1; the interval is multiplied by the difficulty factor and capped at 15 s (from level 7)
- **Bombs**: Instant clear of every enemy currently alive and all enemy bullets (plus the unlocks above). Capacity scales with bomb upgrade level (max 5)

### Level Progression

Score-based advancement with exponentially growing thresholds:

```
increment(i)      = int(LEVEL_SCORE_BASE * LEVEL_SCORE_GROWTH ^ (i - 1))
score_for(level)  = sum of increment(i) for i = 1 .. level - 1     # cumulative score
```

- Level 1 → 2: 500 points
- Level 2 → 3: 1,200 points cumulative (level 4: 2,179; level 5: 3,550; level 10: 24,571; level 20: 745,777)
- Difficulty factor per level: `f = 1.12 ^ (level - 1)`

Difficulty affects:
- Enemy spawn delay: `max(400ms, 1200ms / f)` (floor reached at level 11)
- Asteroid spawn delay: `max(800ms, 2500ms / f)` (floor reached at level 12)
- Missile pickup interval: `min(15000ms, 8000ms * f)`
- Enemy fire delay: `max(600ms, base / f)`
- Enemy speed: `3 + (level - 1) * 0.4` to `6 + (level - 1) * 0.5` px/frame
- Asteroid speed and size: see [Obstacles](#obstacles)
- Shooter probability: table lookup by level
- Background: from level 90 the scenery switches permanently to space (see [Parallax Rendering](#parallax-rendering))

---

## AI / Machine Learning System

### Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                       NeatTrainer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐   │
│  │  NEAT Config │  │  Population  │  │  Checkpoints  │   │
│  │ (neat_config │  │ (150 config, │  │ (every 10     │   │
│  │   .txt)      │  │ 300 headless)│  │  gens, last 2)│   │
│  └──────┬───────┘  └──────┬───────┘  └───────────────┘   │
│         │                 │                              │
│         ▼                 ▼                              │
│  ┌─────────────────────────────────┐                     │
│  │   Multiprocess Evaluation Pool  │                     │
│  │   (headless: cpu_count - 1      │                     │
│  │    workers; visual: serial)     │                     │
│  └──────────────┬──────────────────┘                     │
│                 │                                        │
│                 ▼                                        │
│  ┌─────────────────────────────────┐                     │
│  │          GameEnv                │                     │
│  │  ┌───────────┐  ┌────────────┐  │                     │
│  │  │ Game Loop │  │ Observation│  │                     │
│  │  │ (virtual  │  │ Builder    │  │                     │
│  │  │  time)    │  │ (52-dim)   │  │                     │
│  │  └───────────┘  └────────────┘  │                     │
│  └─────────────────────────────────┘                     │
│                                                          │
│  ┌─────────────────────────────────┐                     │
│  │      Imitation Learning         │                     │
│  │  ┌──────────────────────────┐   │                     │
│  │  │  Human Sessions (.pkl)   │   │                     │
│  │  │  Action similarity bonus │   │                     │
│  │  └──────────────────────────┘   │                     │
│  └─────────────────────────────────┘                     │
└──────────────────────────────────────────────────────────┘
```

`GameEnv` (`src/game_env.py`) re-implements the game loop of `Game._update()` (`src/game.py`) as a steppable environment (`reset(seed)` / `step(actions) -> (obs, reward, done)`) without audio, and with rendering (background + minimal HUD) only when created with `render=True`. The entity classes are shared between the two loops.

### Neural Network I/O

**52 Inputs** (built by `build_observation()` in `src/game_env.py`, shared by `GameEnv`, `HumanRecorder` and the Coop AI ship; positions are divided by the screen size, most other values are scaled to roughly 0-1, missing entities are zero-padded):

| Group | Inputs | Description |
|-------|--------|-------------|
| Player state | 14 | Position (x, y), lives, shield HP, invincibility flag, missiles (capped at 3), bombs (capped at 3), level, shoot cooldown, enemy density, bullet density, weapon/engine/shield upgrade levels |
| 3 nearest enemies | 12 | Relative dx, dy, speed, is_shooter (4 per enemy) |
| 3 nearest asteroids | 12 | Relative dx, dy, speed, radius (4 per asteroid) |
| 3 nearest enemy bullets | 12 | Relative dx, dy, velocity_x, velocity_y (4 per bullet) |
| Nearest power-up | 2 | Relative dx, dy (upgrade power-up or missile pickup) |

**5 Outputs** (decoded identically by `neat_trainer._decode_actions_static` and `Game._decode_ai_actions`):

| Output | Threshold | Action |
|--------|-----------|--------|
| `out[0]` | < -0.2 / > 0.2 | Move up / Move down |
| `out[1]` | < -0.2 / > 0.2 | Move left / Move right |
| `out[2]` | > 0.3 | Shoot |
| `out[3]` | > 0.5 | Fire missile |
| `out[4]` | > 0.5 | Detonate bomb |

### Fitness Function

Fitness is computed by `_compute_fitness(env)` in `src/neat_trainer.py` from the statistics counters of `GameEnv` at the end of the episode. Multi-component fitness with 15 terms (the game score itself is not a term):

```
fitness = enemies_killed × 50          # bullet kills only
        + dodges × 5                   # enemies/asteroids that left the screen
        + accuracy_bonus               # (shots_hit / shots_fired) × enemies_killed × 10
        + frame_count × 0.05           # survival time
        - lives_lost × 200             # death penalty
        + level² × 50                  # progression bonus
        + powerups_collected × 50
        + missiles_collected × 30
        + total_upgrade_levels × 20
        - shield_hits × 15
        - shield_breaks × 120
        - idle_ratio × frame_count × 0.5   # inactivity penalty (frames without movement)
        + near_misses × 3              # sprites that came within 60 px
        + powerup_approach_frames × 0.1
        - shots_fired × 0.01
```

Fitness is not clipped at 0. `shots_fired` counts volleys, `shots_hit` counts bullets that hit at least one enemy. In headless mode, a genome whose evaluation raises an exception gets a fitness of -10000 (with no imitation bonus). The imitation bonus (below) is added on top.

**Per-step reward** (`GameEnv.step()`): `step()` also returns a shaped reward, which `NeatTrainer` does not use (fitness comes only from `_compute_fitness`):
- Survival bonus: `+0.02` per frame
- Bullet kill: `+5` per enemy
- Death penalty: `-200` per life lost
- Shot penalty: `-0.01` per volley fired
- Shield: `-15` per shield hit, `-60` per shield break
- Pickups: `+10` per upgrade power-up, `+5` per missile pickup, `+5` per upgrade level gained
- Level up: `+level² × 5`
- Near-miss bonus: `+3.0` the first time an enemy, asteroid or enemy bullet comes within 60 px of the player
- Power-up approach: `+0.1` per frame getting closer to the nearest power-up

**Early termination** (constants in frames, `EARLY_STOP_*`): from frame 180 (`EARLY_STOP_MIN_FRAME`) the episode is checked whenever the frame index is a multiple of 150 (`EARLY_STOP_INTERVAL`), i.e. at frame 300, 450, 600, … The genome is stopped if it has lost 3 lives (`EARLY_STOP_MAX_LIVES_LOST`; with `PLAYER_LIVES = 3` this coincides with game over) or if it has been idle (no movement) for at least 70% of the frames (`EARLY_STOP_IDLE_RATIO`). In headless workers the lives check applies from generation 10; in visual mode the limit before generation 10 is `PLAYER_LIVES`, i.e. the same value. The source comments assume 30 FPS ("every 5 seconds", "6 seconds"); at the actual `FPS = 60` the check runs every 2.5 s, starting at 5 s of game time.

### Imitation Learning

Human gameplay is recorded during **Play** and **Coop** modes via the `HumanRecorder` class (one frame per game update, human ship only).

**Recording format** (per frame):
```python
(observation_52dim, action_dict)  # action = {vertical, horizontal, shoot, missile, bomb}
```

**Sessions stored**: `ai/human_sessions/session_{timestamp}.pkl`, a dict `{frames, score, num_frames, timestamp}`, written at game over (max 20 retained)

**Bonus application** (from generation 10, `IMITATION_START_GEN`, computed for every genome: in headless mode by the worker process right after the episode, in visual mode in the main process):
1. Load the human sessions once per training run, before the worker pool is created, and use the 5 most recent valid ones, each weighted by `max(1, score / 500)`; only the sampled frames are kept in memory and passed to the workers
2. Sample frames uniformly with step `max(1, len(frames) // 200)` (about 200-400 frames per session)
3. Feed each observation to the evolved network
4. Compare network output with recorded human action
5. Score similarity across all 5 action dimensions: movement axes 1 (match) / 0.5 (off by one) / 0 (opposite), shoot/missile/bomb 1 if equal; average → 0-1
6. Apply bonus: `weighted_avg_similarity × 300` (max 300 fitness points)

This acts as a **soft behavioral prior** — once the bonus is active, genomes that play like humans receive a fitness bonus, steering evolution away from degenerate strategies.

### Training Pipeline

```
1. Load NEAT config + latest checkpoint (if any)
   - a resumed run uses the config stored in the checkpoint
   - headless forces pop_size = 300 (new or resumed run)
   - ai/best_genome.pkl (if present) sets the current fitness record
2. For each generation:
   a. Evaluate all genomes
      - headless: persistent multiprocessing.Pool, imap_unordered (chunksize=1)
      - visual: serial, in-process, rendering 1 frame out of 4
      - Each evaluation: compile the genome's FeedForwardNetwork into a generated Python function
        (same outputs as neat's activate(), about 2x faster) → run GameEnv episode
      - Episode length cap: 900 → 5400 frames, linear over the first 50 generations
        (15 s → 90 s of game time at FPS = 60; the source comments say 30 s - 3 min, assuming 30 FPS)
      - Virtual time system ensures deterministic evaluation
        (pygame.time.get_ticks patched to frame_count × 1000 / FPS, ≈16.7 ms per frame)
      - Per-genome seed: genome_id + generation × 10000
   b. Apply imitation learning bonus (from gen 10; headless: computed in the workers)
   c. Save best genome to ai/best_genome.pkl if new record
   d. Check for stagnation, adjust speciation / inject diversity if needed
   e. Update real-time dashboard
   f. NEAT reproduction + speciation (neat-python)
   g. Checkpoint every 10 generations (only the last 2 kept)
```

**Parallelization**: The `Pool` is created once per training run. Each worker process initializes with `SDL_VIDEODRIVER=dummy`, `SDL_AUDIODRIVER=dummy` and a 1×1 display for headless Pygame operation, receives the imitation samples through the pool initializer, and lowers its priority to `BELOW_NORMAL` on Windows. Genome evaluation is distributed via `pool.imap_unordered()`; the dashboard is refreshed (and `ESC` polled) every 5 completed genomes.

### NEAT Hyperparameters

From `ai/neat_config.txt`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Population size | 150 | Forced to 300 in headless training |
| Fitness criterion | max | `fitness_threshold = 99999`, `no_fitness_termination = True`, `reset_on_extinction = True` |
| Initial topology | 52→5 fully connected (`full_direct`) | No hidden nodes, feed-forward |
| Activation | tanh (default) | sigmoid, relu available via mutation (rate 10%) |
| Aggregation | sum | Not mutated |
| Weight mutation rate | 40% | Power 0.5, replace rate 10%, range ±5 |
| Bias mutation rate | 70% | Power 0.5, replace rate 10%, range ±5 |
| Connection add prob | 40% | Structural growth |
| Node add prob | 35% | Hidden neuron insertion |
| Connection delete prob | 20% | Pruning |
| Node delete prob | 10% | Simplification |
| Compatibility threshold | 3.0 | Initial species boundary (adjusted at runtime by anti-stagnation) |
| Compatibility coefficients | disjoint 1.0, weight 0.5 | |
| Stagnation limit | 12 gens | Per-species patience (`species_fitness_func = max`) |
| Species elitism | 2 | Species protected from stagnation removal |
| Elitism | 3 | Top genomes preserved per species |
| Survival threshold | 20% | Only top 20% reproduce |
| Min species size | 3 | Prevents tiny species |

### Anti-Stagnation Mechanisms

Run by `_check_and_fix_stagnation()` after every generation (the fitness history is kept in memory and restarts with each training run):

1. **Speciation control**: If 2 or fewer species remain, the compatibility threshold is multiplied by 0.85 (min 1.0) to encourage speciation; with more than 20 species it is multiplied by 1.1 (max 6.0) to avoid fragmentation
2. **Fitness plateau detection**: If the best-of-generation fitness varies by less than 3% over the last 20 generations, the worst 15% of the population (min 10 genomes) is replaced: half with fresh random genomes, half with copies of the all-time best genome mutated twice
3. **Deep stagnation**: If it varies by less than 5% over the last 50 generations, an aggressive injection replaces 25% (min 10), with best-genome copies mutated three times
4. **Species stagnation** (neat-python): species that don't improve for 12 generations are removed (the 2 best species are always protected)

Injection happens inside the evaluation callback, after the generation has been evaluated, and the injected genomes have `fitness = None`. neat-python 1.1.0's `Population.run()` raises `RuntimeError("Fitness not assigned to genome …")` when any genome lacks a fitness after evaluation.

### Checkpointing & Persistence

| File | Format | Content |
|------|--------|---------|
| `ai/neat-checkpoint-{N}` | gzip-compressed pickle | Full NEAT state: generation, config, population, species, random state |
| `ai/best_genome.pkl` | Pickle | Single best-performing genome (used for Coop mode) |
| `ai/human_sessions/session_*.pkl` | Pickle | Recorded human (observation, action) frame pairs + final score |
| `scores.json` | JSON | Top-10 human player leaderboard (`name`, `score`) |

Checkpoints are saved every 10 generations by `_RollingCheckpointer`, which deletes all but the 2 most recent (`MAX_CHECKPOINTS = 2`), and are automatically discovered on startup (highest generation loaded for resume). `ai/best_genome.pkl` is overwritten whenever a generation beats the record. These files are committed to the repository so training can continue across machines.

---

## Project Structure

```
Yellow-Star/
├── main.py                  # Entry point — initializes Pygame, runs menu loop
├── settings.py              # All game constants and configuration values
├── requirements.txt         # Python dependencies
├── run.bat                  # Windows launcher (activates venv + runs main.py)
├── scores.json              # Persistent leaderboard (top 10)
├── generate_enemies.py      # Utility: procedurally generates 16 enemy sprites (overwrites the PNGs)
├── CLAUDE.md                # Guidance for Claude Code (commands, architecture notes)
│
├── src/
│   ├── __init__.py
│   ├── game.py              # Core game loop, collision detection, HUD rendering, coop AI
│   ├── player.py            # Player sprite, movement, shooting, shield/bomb logic
│   ├── enemy.py             # Enemy sprites, 4 shooting behaviors
│   ├── bullet.py            # Player bullet sprite with animated flame trail
│   ├── enemy_bullet.py      # Enemy projectile with angle-based movement
│   ├── missile.py           # Piercing missile + missile power-up collectible
│   ├── asteroid.py          # Procedural polygon asteroid
│   ├── explosion.py         # 7-frame procedural explosion animation
│   ├── powerup.py           # 4 power-up types (Engine, Weapon, Shield, Bomb)
│   ├── upgrades.py          # UpgradeManager — 4 categories × 10 levels
│   ├── level.py             # LevelManager — score thresholds, difficulty scaling
│   ├── menu.py              # Main menu with 4 mode options
│   ├── parallax.py          # Biome-based parallax background (6 surface biomes + space)
│   ├── scores.py            # JSON leaderboard persistence
│   ├── sounds.py            # Procedural audio synthesis (SFX + BGM)
│   ├── ship_sprites.py      # Player ship sprite loading (10 variants, one per weapon level)
│   ├── game_env.py          # GameEnv wrapper for NEAT — headless stepping, observation builder
│   ├── neat_trainer.py      # NeatTrainer — evolution loop, multiprocessing, dashboard
│   └── human_recorder.py    # Records human gameplay sessions for imitation learning
│
├── ai/
│   ├── neat_config.txt      # NEAT hyperparameter configuration
│   ├── best_genome.pkl      # Best evolved genome (for Coop mode)
│   ├── neat-checkpoint-*    # Population checkpoints (every 10 generations, last 2 kept)
│   └── human_sessions/      # Recorded human play sessions (.pkl)
│
└── assets/
    └── images/
        ├── player/          # 10 player ship sprites (ship_01.png - ship_10.png)
        └── enemies/         # 16 enemy ship sprites (enemy_01.png - enemy_16.png)
```

---

## Configuration Reference

All game constants are defined in [settings.py](settings.py) (NEAT parameters are in `ai/neat_config.txt`; training constants such as `MAX_FRAMES_PER_EPISODE` and `EARLY_STOP_*` are at the top of `src/neat_trainer.py`). Speeds are in px per frame at 60 FPS:

### Display

| Constant | Value | Description |
|----------|-------|-------------|
| `SCREEN_WIDTH` | 1280 | Window width in pixels |
| `SCREEN_HEIGHT` | 740 | Window height in pixels |
| `FPS` | 60 | Target frame rate (also the base of the training virtual clock) |

### Player

| Constant | Value | Description |
|----------|-------|-------------|
| `PLAYER_SPEED` | 6 | Base movement speed (px/frame) |
| `PLAYER_SIZE` | (96, 40) | Imported but not used: the ship size comes from `ship_sprites.py` (58-130 px wide) |
| `PLAYER_SHOOT_DELAY` | 250 | Base milliseconds between shots |
| `PLAYER_LIVES` | 3 | Starting lives |
| `PLAYER_INVINCIBLE_TIME` | 2000 | Base post-hit invincibility (ms) |

### Enemies

| Constant | Value | Description |
|----------|-------|-------------|
| `ENEMY_SPEED_MIN / MAX` | 3 / 6 | Base speed range (px/frame) |
| `ENEMY_MODELS` | 16 | Number of unique enemy sprites |
| `ENEMY_SPAWN_DELAY` | 1200 | Base spawn interval (ms) |
| `ENEMY_SHOOT_DELAY_*` | 1400-3200 | Base per-type fire rates (ms) |
| `ENEMY_BURST_ANGLES` | (-12, 0, 12) | Burst spread (degrees) |
| `ENEMY_SHOOTER_CHANCE` | 0.00 → 0.62 | Shooter probability per level (20 entries, last one reused) |
| `ENEMY_SHOOTER_UNLOCK` | basic 3, fast 4, aimed 6, burst 10 | Unlock level per shooter type |

### Projectiles & Weapons

| Constant | Value | Description |
|----------|-------|-------------|
| `BULLET_SPEED` | 12 | Player bullet speed (px/frame) |
| `ENEMY_BULLET_SPEED` | 7 | Enemy bullet speed (px/frame) |
| `MISSILE_SPEED` | 14 | Missile speed (px/frame) |
| `MISSILE_KILL_COUNT` | 5 | Max enemies destroyed per missile |
| `MISSILE_POWERUP_SPAWN_DELAY` | 8000 | Base missile pickup interval (ms), capped at 15000 by `LevelManager` |

### Difficulty Scaling

| Constant | Value | Description |
|----------|-------|-------------|
| `LEVEL_SCORE_BASE` | 500 | Points for level 1 → 2 |
| `LEVEL_SCORE_GROWTH` | 1.4 | Exponential multiplier of each successive threshold increment |
| `LEVEL_DIFFICULTY_FACTOR` | 1.12 | Per-level difficulty multiplier |
| `LEVEL_MAX` | 100 | Maximum reachable level |

### Power-ups

| Constant | Value | Description |
|----------|-------|-------------|
| `POWERUP_DROP_CHANCE` | 0.15 | 15% drop on enemy kill (bullet or missile) |
| `POWERUP_SPEED` | 2 | Drift speed (px/frame) |
| `BOMB_FREEZE_DURATION` | 1000 | Spawn freeze on bomb (ms) |
| `BOMB_FIELD_DURATION` | 2000 | Protective field on lv10 bomb (ms) |

---

## Technical Details

### Procedural Audio Synthesis

The `SoundManager` generates **all audio at runtime** using waveform math (44.1 kHz, 16 mixer channels) — no external audio files are needed. Audio is used only in Play and Coop modes, not during training.

**Sound effects** (10 unique effects):

| Effect | Duration | Synthesis Technique |
|--------|----------|-------------------|
| Shoot | 0.12s | Frequency sweep 1200→480 Hz + square wave |
| Explosion | 0.45s | Filtered noise + 60 Hz sub-bass boom + 200→83 Hz mid tone |
| Power-up | 0.35s | 4-note ascending arpeggio (C5-E5-G5-C6) |
| Player Hit | 0.3s | Impact noise + 400→100 Hz descending tone |
| Game Over | 1.2s | 3 descending notes (E4-C4-A3), triangle + sine |
| Missile Launch | 0.25s | 200→950 Hz rising tone + filtered noise |
| Bomb | 0.8s | Dual sub-bass (40+80 Hz) + crunch noise |
| Level Up | 0.5s | 5-note fanfare (C5-E5-G5-C6-E6) |
| Shield Hit | 0.2s | Metallic harmonics (800+1600+2400 Hz) + 3200 Hz initial ting |
| Shield Break | 0.5s | Glass noise + 1200→200 Hz sweep |

**Background music**: ~17.5-second loop (8 bars at 110 BPM) in Am (Am-F-C-G, played twice), looping on a dedicated channel and paused at game over. Composed of 5 layers: pulsing bass, chord pad, arpeggio, hi-hat, and kick drum — all synthesized from sine and square waveforms.

### Parallax Rendering

The background (`ParallaxBackground` in `src/parallax.py`) is a **biome system**. Each biome is a `Biome` subclass that, when created, builds a static sky surface (gradient with sun, moon, stars, aurora or light rays baked in) plus a stack of scrolling layers drawn back-to-front:

| Biome | Sky | Scrolling layers (speed) |
|-------|-----|--------------------------|
| Mountain | Blue gradient, sun with glow | 3 midpoint-displacement ridges with snow caps and haze (0.3 / 0.6 / 1.0), conifers (1.8), rocky ground (2.5) |
| Sea | Blue → sand gradient, sun | Clouds (0.3), islands (0.6), water with sun glitter (1.0), 2 wave bands (1.8 / 2.5) |
| City | Night gradient, stars, moon | 3 skyline bands with lit windows and antennas (0.3 / 0.6 / 1.0), landmark silhouettes with warm outline — Eiffel Tower, Big Ben, pyramid, pagoda, Colosseum, Empire State Building (1.0), road with street lamps (2.5) |
| Jungle | Teal → cream gradient, light rays | 2 misty hills (0.3 / 0.6), canopy (1.0), palms and vines (1.8), undergrowth (2.5) |
| Desert | Sunset gradient, sun | Mesas (0.3), 2 dune ridges (0.6 / 1.0), cacti (1.8), sand (2.5) |
| Arctic | Night gradient, stars, aurora | 2 icy ridges with snow (0.3 / 0.6), icebergs (1.0), cracked ice floor (2.5) |
| Space | Dark gradient | Nebulae (0.15), 3 star fields (0.3 / 0.6 / 1.0), planets with shadow and optional rings (0.72), debris (2.5) |

Layer speeds come from the `PARALLAX_*_SPEED` constants (0.3 / 0.6 / 1.0 / 1.8 / 2.5 px/frame); the space biome also uses fractions of them.

- **Rotation**: the 6 surface biomes (`SURFACE_BIOMES`) play in a shuffled order, reshuffled when the queue runs out (the current biome is not repeated immediately). Every biome instance is regenerated with a new seed from the background's own `random.Random`, so each visit looks different.
- **Crossfade**: after `BIOME_DURATION` (60000 ms) in a biome, the next one fades in over `BIOME_FADE_DURATION` (3000 ms): it is drawn on an alpha surface whose opacity rises from 0 to 255 while both biomes keep scrolling. The biome timer advances by `dt × 1000 / 60` ms per update. The next surface biome is built in a background thread `BIOME_PREBUILD_LEAD` (5000 ms) before the switch, with the same class and seed a synchronous build would use, so the switch frame does not stall.
- **Space**: once the level reaches `BIOME_SPACE_LEVEL` (90), the background fades to `SpaceBiome` and stays there.
- `Game` passes the current level to `update()`. `GameEnv` creates the background only with `render=True` (visual training) and calls `update(dt)` without a level, so the space biome never appears there; headless training has no background.

Each layer is a `ScrollingLayer` tile `2 × SCREEN_WIDTH` wide with seamless wrapping via double-blit; surface-biome layers are cropped at build time to the rows that contain non-transparent pixels (same image, less per-pixel alpha blending); ground strips start at `0.82 × SCREEN_HEIGHT` (the sea surface at `0.62 × SCREEN_HEIGHT`). The color constants `SKY_BLUE`, `LIGHT_BLUE`, `MOUNTAIN_FAR/MID/NEAR` and `GROUND_COLOR` in `settings.py` are not used by the biome renderer.

### Procedural Sprite Generation

**Enemy sprites** (`generate_enemies.py`): 16 unique pixel-art ships drawn on a 20×28 logical grid (3x physical pixel scale), then rotated 90° to face left and saved as PNG (84×60 px). Each design has distinct fuselage, cockpit, wings, and engine components.

**Explosions** (`src/explosion.py`): 7-frame cached animation on a 16×16 grid (3x scale), each frame shown for 3 ticks, with a 10-color palette progression: white flash → yellow/orange expansion → red contraction → smoke dispersal.

**Asteroids** (`src/asteroid.py`): Irregular polygons with 7-10 vertices, brown/tan coloring with a darker outline, and 1-3 impact craters. The sprite is not rotated (a `rotation_speed` value is assigned but never applied).

Player ships are not procedural: `src/ship_sprites.py` loads `ship_01.png`-`ship_10.png` (one per weapon level, level 0 uses `ship_01`) and scales them to widths from 58 to 130 px.

---

## Requirements

- **Python** >= 3.10
- **OS**: Windows (tested), Linux/macOS (should work with Pygame-CE)
- **Hardware**: Multi-core CPU recommended for headless training (uses `cpu_count - 1` workers)

```
pygame-ce>=2.5.0
neat-python
```

Local development environment (`venv/`, not versioned): Python 3.14.0, pygame-ce 2.5.6, neat-python 1.1.0.

---

## License

This project is provided as-is for educational and research purposes.
