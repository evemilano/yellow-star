# Yellow Star - AI Edition

A side-scrolling space shooter built with **Pygame-CE** featuring a full **NEAT neuroevolution** pipeline that evolves neural networks to play the game autonomously. Includes human demonstration recording, imitation learning, cooperative human+AI gameplay, and a real-time training dashboard.

---

## Table of Contents

- [Features](#features)
- [Screenshots & Modes](#screenshots--modes)
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
- **Multiprocess Training**: Parallel genome evaluation across all CPU cores
- **Real-Time Dashboard**: Live fitness metrics, species breakdown, and behavior statistics during training
- **Procedural Audio**: All sound effects and background music generated at runtime via waveform synthesis (zero external audio files)
- **Procedural Sprites**: Enemy ships and explosions are generated programmatically
- **Upgrade System**: 4 upgrade categories (Engine, Weapon, Shield, Bomb) with 10 levels each
- **16 Enemy Types**: Unique pixel-art ships with 4 distinct shooting behaviors
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
| `neat-python` | latest | NEAT neuroevolution library |
| Python | >= 3.10 | Runtime |

---

## Quick Start

```bash
# Run the game (activates venv automatically on Windows)
run.bat

# Or manually
python main.py
```

The main menu presents four options navigable with arrow keys:

| Key | Action |
|-----|--------|
| `UP` / `DOWN` | Navigate menu |
| `ENTER` | Select option |
| `ESC` | Quit |

---

## Game Modes

### 1. Play (Human)
Standard keyboard-controlled gameplay. All sessions are **automatically recorded** to `ai/human_sessions/` for later use in imitation learning. Up to 20 sessions are retained.

### 2. Train AI (Headless)
Background NEAT training with no game rendering. Uses **multiprocessing** to evaluate genomes in parallel across all CPU cores. Displays a real-time terminal dashboard with fitness metrics, species info, and efficiency statistics.

### 3. View AI Training (Visual)
Same NEAT training but with live rendering of each genome playing. Useful for observing evolved behaviors. Runs single-threaded at reduced speed.

### 4. Coop AI
Human and a trained AI agent play **simultaneously** on the same screen. Requires a saved `ai/best_genome.pkl` from prior training. Both players share the game world — enemies, power-ups, and score are pooled.

---

## Gameplay Mechanics

### Player Controls

| Key | Action |
|-----|--------|
| `Arrow Keys` | Move (4 directions) |
| `SPACE` | Shoot |
| `M` | Fire homing missile |
| `B` | Detonate bomb |

- **Speed**: Base 6 px/frame, scales up to 2x with engine upgrades
- **Health**: 3 lives + shield HP from upgrades
- **Invincibility**: 2–4 seconds after being hit (scales with shield level)

### Enemies

16 unique pixel-art enemy models spawn from the right edge and scroll left. Four shooting behaviors unlock progressively:

| Type | Unlock Level | Fire Delay | Behavior |
|------|-------------|------------|----------|
| **Basic** | 3 | 2200 ms | Straight shot leftward |
| **Fast** | 4 | 1400 ms | Straight shot, high frequency |
| **Aimed** | 6 | 2800 ms | Calculates angle to player |
| **Burst** | 10 | 3200 ms | 3-bullet spread (-12, 0, +12 degrees) |

Shooter probability per enemy scales from 0% (level 1-2) to 62% (level 20), interpolated linearly between defined checkpoints.

### Obstacles

**Asteroids** are procedurally generated irregular polygons (7-10 vertices) with rotation animation. Size ranges from 20-45 px radius at level 1, scaling up to 40-65 px at higher levels.

### Power-ups & Upgrades

Killed enemies have a **15% drop chance** for one of 4 power-up types:

| Type | Color | Icon | Effect |
|------|-------|------|--------|
| **Engine** | Green | M | +10% speed per level (max 2x) |
| **Weapon** | Orange | W | +bullets and -fire delay |
| **Shield** | Blue | S | +shield HP, +invincibility duration |
| **Bomb** | Purple | B | +bomb capacity, unlock effects |

Each category has **10 upgrade levels**. Key milestones:

**Weapon progression:**
| Level | Bullets | Fire Delay |
|-------|---------|-----------|
| 0-1 | 1 | Base (250ms) |
| 2-3 | 2 | -15% |
| 4-5 | 3 | -15% |
| 6-7 | 4 | -15% |
| 8-10 | 5 | -30% at lv10 |

**Bomb unlock chain:**
| Level | Effect |
|-------|--------|
| 2+ | Destroys asteroids |
| 6+ | Double score on kills (200 pts) |
| 8+ | Freezes spawns for 1 second |
| 10 | Shield field for 2 seconds (full invincibility) |

### Weapons

- **Bullets**: 12 px/frame, animated 3-frame sprites with flame trail. Score: 100 pts/kill
- **Missiles**: Homing projectiles (14 px/frame) that auto-target up to 5 enemies. Score: 150 pts/kill. Collected as floating power-ups (8s base spawn rate)
- **Bombs**: Area-of-effect instant clear of all on-screen enemies. Capacity scales with bomb upgrade level (max 5)

### Level Progression

Score-based advancement with exponential scaling:

```
threshold(level) = LEVEL_SCORE_BASE * LEVEL_SCORE_GROWTH ^ (level - 1)
```

- Level 1 → 2: 500 points
- Level 2 → 3: ~1,200 points cumulative
- Difficulty factor per level: `1.12 ^ (level - 1)`

Difficulty affects:
- Enemy spawn delay: `max(400ms, 1200ms / factor)`
- Asteroid spawn delay: `max(800ms, 2500ms / factor)`
- Enemy speed: `base + level * 0.4` to `base + level * 0.5`
- Shooter probability: scales with level index

---

## AI / Machine Learning System

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    NeatTrainer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  NEAT Config  │  │  Population  │  │  Checkpoints │  │
│  │  (neat_config │  │  (100 genomes│  │  (every 10   │  │
│  │   .txt)       │  │   per gen)   │  │   gens)      │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘  │
│         │                 │                              │
│         ▼                 ▼                              │
│  ┌─────────────────────────────────┐                    │
│  │   Multiprocess Evaluation Pool  │                    │
│  │   (cpu_count - 1 workers)       │                    │
│  └──────────────┬──────────────────┘                    │
│                 │                                        │
│                 ▼                                        │
│  ┌─────────────────────────────────┐                    │
│  │          GameEnv                │                    │
│  │  ┌──────────┐  ┌────────────┐  │                    │
│  │  │ Game Loop │  │ Observation│  │                    │
│  │  │ (headless)│  │ Builder    │  │                    │
│  │  └──────────┘  │ (52-dim)   │  │                    │
│  │                 └────────────┘  │                    │
│  └─────────────────────────────────┘                    │
│                                                          │
│  ┌─────────────────────────────────┐                    │
│  │      Imitation Learning         │                    │
│  │  ┌──────────────────────────┐   │                    │
│  │  │  Human Sessions (.pkl)   │   │                    │
│  │  │  Action similarity bonus │   │                    │
│  │  └──────────────────────────┘   │                    │
│  └─────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────┘
```

### Neural Network I/O

**52 Inputs (normalized to 0-1 or -1 to 1):**

| Group | Inputs | Description |
|-------|--------|-------------|
| Player state | 14 | Position (x, y), lives, shield HP, invincibility flag, missiles, bombs, level, shoot cooldown, enemy density, bullet density, weapon/engine/shield upgrade levels |
| 3 nearest enemies | 12 | Relative dx, dy, speed, is_shooter (4 per enemy) |
| 3 nearest asteroids | 12 | Relative dx, dy, speed, radius (4 per asteroid) |
| 3 nearest enemy bullets | 12 | Relative dx, dy, velocity_x, velocity_y (4 per bullet) |
| Nearest power-up | 2 | Relative dx, dy |

**5 Outputs:**

| Output | Threshold | Action |
|--------|-----------|--------|
| `out[0]` | < -0.2 / > 0.2 | Move up / Move down |
| `out[1]` | < -0.2 / > 0.2 | Move left / Move right |
| `out[2]` | > 0.0 | Shoot |
| `out[3]` | > 0.5 | Fire missile |
| `out[4]` | > 0.5 | Detonate bomb |

### Fitness Function

Multi-component fitness with 14 factors:

```
fitness = score × 2.0
        + enemies_killed × 30
        + dodges × 5
        + accuracy_bonus              # (hit% × kills × 10)
        + frame_count × 0.05          # survival time
        - lives_lost × 200            # death penalty
        + level × 100                 # progression bonus
        + powerups_collected × 50
        + missiles_collected × 30
        + total_upgrade_levels × 20
        - shield_hits × 30
        - shield_breaks × 80
        - idle_ratio × frames × 0.5   # inactivity penalty
        + near_misses × 3             # close call bonus
        + powerup_approach × 0.1
```

**Reward shaping per step:**
- Survival bonus: `+0.02` per frame
- Death penalty: `-200` per life lost
- Near-miss bonus: `+2.0` for dodging within 60px
- Position bonus: `+0.01` if `x > 200` (discourages wall-hugging)
- Power-up approach: `+0.5` per frame getting closer to nearest power-up

**Early termination** (after gen 10): genome killed if 2+ lives lost or 95%+ idle time. Checked every 150 frames.

### Imitation Learning

Human gameplay is recorded during **Play** and **Coop** modes via the `HumanRecorder` class.

**Recording format** (per frame):
```python
(observation_52dim, action_dict)  # action = {vertical, horizontal, shoot, missile, bomb}
```

**Sessions stored**: `ai/human_sessions/session_{timestamp}.pkl` (max 20 retained)

**Bonus application** (after generation 10):
1. Load last 5 human sessions, weighted by final score
2. Sample up to 200 frames per session
3. Feed each observation to the evolved network
4. Compare network output with recorded human action
5. Score similarity across all 5 action dimensions (0-1 per dimension)
6. Apply bonus: `similarity_ratio × weighted_avg × 300` (max ~300 fitness points)

This acts as a **soft behavioral prior** — genomes that play like humans receive a fitness bonus, steering early evolution away from degenerate strategies.

### Training Pipeline

```
1. Load NEAT config + latest checkpoint (if any)
2. For each generation:
   a. Evaluate all 100 genomes in parallel (multiprocessing.Pool)
      - Each worker: create FeedForwardNetwork → run GameEnv episode
      - Episode length: 900-5400 frames (30s-3min, scales with generation)
      - Virtual time system ensures deterministic evaluation
      - Per-genome seed: genome_id + generation × 10000
   b. Apply imitation learning bonus (after gen 10)
   c. Update fitness, run NEAT reproduction + speciation
   d. Save best genome if new record
   e. Checkpoint every 10 generations
   f. Check for stagnation, inject diversity if needed
   g. Update real-time dashboard
```

**Parallelization**: Each worker process initializes with `SDL_VIDEODRIVER=dummy` for headless Pygame operation. Genome evaluation is distributed via `pool.map()`.

### NEAT Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Population size | 100 | Genomes per generation |
| Initial topology | 52→5 fully connected | No hidden nodes |
| Activation | tanh (default) | sigmoid, relu available via mutation |
| Weight mutation rate | 80% | Aggressive parametric search |
| Weight mutation power | 0.5 | Moderate perturbation |
| Connection add prob | 30% | Structural growth |
| Node add prob | 15% | Hidden neuron insertion |
| Connection delete prob | 15% | Pruning |
| Node delete prob | 10% | Simplification |
| Compatibility threshold | 1.5 | Species boundary |
| Stagnation limit | 20 gens | Per-species patience |
| Elitism | 3 | Top genomes preserved |
| Survival threshold | 20% | Only top 20% reproduce |
| Min species size | 2 | Prevents singleton species |

### Anti-Stagnation Mechanisms

1. **Single-species detection**: If only 1 species remains, compatibility threshold is reduced by 10% (min 0.5) to encourage speciation
2. **Fitness plateau detection**: If average fitness improves less than 5% over 30 generations, the worst 10% of the population is replaced with fresh random genomes
3. **Species stagnation**: Individual species that don't improve for 20 generations are eligible for removal (1 elite species always protected)

### Checkpointing & Persistence

| File | Format | Content |
|------|--------|---------|
| `ai/neat-checkpoint-{N}` | Pickle | Full NEAT population state (all genomes, species, stats) |
| `ai/best_genome.pkl` | Pickle | Single best-performing genome (used for Coop mode) |
| `ai/human_sessions/session_*.pkl` | Pickle | Recorded human (observation, action) frame pairs |
| `scores.json` | JSON | Top-10 human player leaderboard |

Checkpoints are saved every 10 generations and automatically discovered on startup (highest generation loaded for resume).

---

## Project Structure

```
Yellow-Star/
├── main.py                  # Entry point — initializes Pygame, runs menu loop
├── settings.py              # All game constants and configuration values
├── requirements.txt         # Python dependencies
├── run.bat                  # Windows launcher (activates venv + runs main.py)
├── scores.json              # Persistent leaderboard (top 10)
├── generate_enemies.py      # Utility: procedurally generates 16 enemy sprites
│
├── src/
│   ├── __init__.py
│   ├── game.py              # Core game loop, collision detection, HUD rendering
│   ├── player.py            # Player sprite, movement, shooting, shield/bomb logic
│   ├── enemy.py             # Enemy sprites, 4 shooting behaviors, level scaling
│   ├── bullet.py            # Player bullet sprite with animated flame trail
│   ├── enemy_bullet.py      # Enemy projectile with angle-based movement
│   ├── missile.py           # Homing missile + missile power-up collectible
│   ├── asteroid.py          # Procedural polygon asteroid with rotation
│   ├── explosion.py         # 7-frame procedural explosion animation
│   ├── powerup.py           # 4 power-up types (Engine, Weapon, Shield, Bomb)
│   ├── upgrades.py          # UpgradeManager — 4 categories × 10 levels
│   ├── level.py             # LevelManager — score thresholds, difficulty scaling
│   ├── menu.py              # Main menu with 4 mode options
│   ├── parallax.py          # 6-layer scrolling background
│   ├── scores.py            # JSON leaderboard persistence
│   ├── sounds.py            # Procedural audio synthesis (SFX + BGM)
│   ├── ship_sprites.py      # Player ship sprite loading (10 variants)
│   ├── game_env.py          # GameEnv wrapper for NEAT — headless stepping, observation builder
│   ├── neat_trainer.py       # NeatTrainer — evolution loop, multiprocessing, dashboard
│   └── human_recorder.py    # Records human gameplay sessions for imitation learning
│
├── ai/
│   ├── neat_config.txt      # NEAT hyperparameter configuration
│   ├── best_genome.pkl      # Best evolved genome (for Coop mode)
│   ├── neat-checkpoint-*    # Population checkpoints (every 10 generations)
│   └── human_sessions/      # Recorded human play sessions (.pkl)
│
└── assets/
    └── images/
        ├── player/          # 10 player ship sprites (ship_01.png - ship_10.png)
        └── enemies/         # 16 enemy ship sprites (enemy_01.png - enemy_16.png)
```

---

## Configuration Reference

All constants are defined in [settings.py](settings.py):

### Display

| Constant | Value | Description |
|----------|-------|-------------|
| `SCREEN_WIDTH` | 1280 | Window width in pixels |
| `SCREEN_HEIGHT` | 900 | Window height in pixels |
| `FPS` | 30 | Target frame rate |

### Player

| Constant | Value | Description |
|----------|-------|-------------|
| `PLAYER_SPEED` | 6 | Base movement speed (px/frame) |
| `PLAYER_SIZE` | (96, 40) | Sprite dimensions |
| `PLAYER_SHOOT_DELAY` | 250 | Milliseconds between shots |
| `PLAYER_LIVES` | 3 | Starting lives |
| `PLAYER_INVINCIBLE_TIME` | 2000 | Post-hit invincibility (ms) |

### Enemies

| Constant | Value | Description |
|----------|-------|-------------|
| `ENEMY_SPEED_MIN / MAX` | 3 / 6 | Speed range (px/frame) |
| `ENEMY_MODELS` | 16 | Number of unique enemy sprites |
| `ENEMY_SPAWN_DELAY` | 1200 | Base spawn interval (ms) |
| `ENEMY_SHOOT_DELAY_*` | 1400-3200 | Per-type fire rates (ms) |
| `ENEMY_BURST_ANGLES` | (-12, 0, 12) | Burst spread (degrees) |

### Projectiles & Weapons

| Constant | Value | Description |
|----------|-------|-------------|
| `BULLET_SPEED` | 12 | Player bullet speed (px/frame) |
| `ENEMY_BULLET_SPEED` | 7 | Enemy bullet speed (px/frame) |
| `MISSILE_SPEED` | 14 | Homing missile speed (px/frame) |
| `MISSILE_KILL_COUNT` | 5 | Max enemies per missile |
| `MISSILE_POWERUP_SPAWN_DELAY` | 8000 | Missile pickup interval (ms) |

### Difficulty Scaling

| Constant | Value | Description |
|----------|-------|-------------|
| `LEVEL_SCORE_BASE` | 500 | Points for level 1 → 2 |
| `LEVEL_SCORE_GROWTH` | 1.4 | Exponential threshold multiplier |
| `LEVEL_DIFFICULTY_FACTOR` | 1.12 | Per-level difficulty multiplier |
| `LEVEL_MAX` | 100 | Maximum reachable level |

### Power-ups

| Constant | Value | Description |
|----------|-------|-------------|
| `POWERUP_DROP_CHANCE` | 0.15 | 15% drop on enemy kill |
| `POWERUP_SPEED` | 2 | Drift speed (px/frame) |
| `BOMB_FREEZE_DURATION` | 1000 | Spawn freeze on bomb (ms) |
| `BOMB_FIELD_DURATION` | 2000 | Shield field on lv10 bomb (ms) |

---

## Technical Details

### Procedural Audio Synthesis

The `SoundManager` generates **all audio at runtime** using waveform math — no external audio files are needed.

**Sound effects** (10 unique effects):

| Effect | Duration | Synthesis Technique |
|--------|----------|-------------------|
| Shoot | 0.12s | Frequency sweep 1200→400 Hz + square wave |
| Explosion | 0.45s | Filtered noise + 60 Hz sub-bass boom |
| Power-up | 0.35s | 4-note ascending arpeggio (C5-E5-G5-C6) |
| Player Hit | 0.3s | Impact noise + 400→100 Hz descending tone |
| Game Over | 1.2s | 3-note minor chord (E4-C4-A3) |
| Missile Launch | 0.25s | 200→3200 Hz ascending chirp |
| Bomb | 0.8s | Dual sub-bass (40+80 Hz) + crunch noise |
| Level Up | 0.5s | 5-note fanfare (C5-E5-G5-C6-E6) |
| Shield Hit | 0.2s | Metallic harmonics (800+1600+2400 Hz) |
| Shield Break | 0.5s | Glass noise + 1200→200 Hz sweep |

**Background music**: ~16-second loop at 110 BPM in Am (Am-F-C-G). Composed of 5 layers: sub-bass, chord pad, arpeggio, hi-hat, and kick drum — all synthesized from sine/noise generators.

### Parallax Rendering

6-layer scrolling background rendered back-to-front:

| Layer | Speed | Content |
|-------|-------|---------|
| Sky | Static | Gradient fill (sky blue → light blue) |
| Mountains Far | 0.3x | 12-segment jagged peaks |
| Mountains Mid | 0.6x | 16-segment peaks |
| Mountains Near | 1.0x | 20-segment hills |
| Trees | 1.8x | 40 procedural trunk+canopy pairs |
| Ground | 2.5x | Wavy terrain strip |

Each layer uses `ScrollingLayer` with seamless tile wrapping via double-blit.

### Procedural Sprite Generation

**Enemy sprites** (`generate_enemies.py`): 16 unique pixel-art ships drawn on a 20×28 logical grid (3x physical pixel scale), then rotated 90° to face left. Each design has distinct fuselage, cockpit, wings, and engine components.

**Explosions** (`src/explosion.py`): 7-frame cached animation on a 16×16 grid with 8-color palette progression: white flash → yellow/orange expansion → red contraction → smoke dispersal.

**Asteroids** (`src/asteroid.py`): Irregular polygons with 7-10 vertices, brown/tan coloring, 1-3 impact craters, and constant rotation.

---

## Requirements

- **Python** >= 3.10
- **OS**: Windows (tested), Linux/macOS (should work with Pygame-CE)
- **Hardware**: Multi-core CPU recommended for headless training (uses `cpu_count - 1` workers)

```
pygame-ce>=2.5.0
neat-python
```

---

## License

This project is provided as-is for educational and research purposes.
