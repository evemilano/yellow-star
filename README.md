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
  - [Training Dashboard](#training-dashboard)
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
- **Training Dashboard**: An HTML page (opened in the browser when training starts) with the full training history: validation record, population trends, gameplay and behavior metrics, fitness breakdown, species, network complexity, explanations of every metric; plus a summary screen in the game window
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
Background NEAT training with no game rendering. Uses **multiprocessing** to evaluate genomes (and to validate the best ones) in parallel on `cpu_count - 1` worker processes; the population size is forced to **200** (`HEADLESS_POP_SIZE`). The game window shows a summary screen (phase and progress, validation record, how the record plays, population fitness trend, species, fitness chart) and the detailed dashboard opens in the browser (see Training Dashboard); neat-python's `StdOutReporter` also prints per-generation statistics to the console. `D` reopens the dashboard; `ESC` or closing the window interrupts training and returns to the menu.

### 3. View AI Training (Visual)
Same NEAT training, but genomes are evaluated serially in the main process with live rendering of each genome playing (biome background, minimal HUD, training overlay). Only 1 frame out of 4 is drawn (`TRAIN_RENDER_SKIP = 4`) and there is no frame-rate cap. Each genome plays the same episodes as in headless mode (same seeds, same fitness). The population size comes from the NEAT config (150 in `ai/neat_config.txt`, also when resuming a headless checkpoint: its first generation still evaluates the restored population). The end-of-generation validation runs in the main process without rendering, with a progress screen.

### 4. Coop AI
Human and a trained AI agent play **simultaneously** on the same screen. Requires a saved `ai/best_genome.pkl` from prior training (otherwise the menu entry is greyed out and `ENTER` on it does nothing). If the file exists but cannot be used (corrupted pickle, or a genome incompatible with `ai/neat_config.txt` and the 52-value observation), the error is printed to the console and the game starts without the AI ship, showing "Genoma AI non valido: si gioca senza AI" for 5 seconds. Both ships share the game world and the score; each ship has its own lives, upgrades, missiles and bombs, and power-ups go to whichever ship picks them up. The AI ship is tinted blue with an "AI" label. When the AI loses its 3 lives it is removed ("AI eliminata!"); the game ends only when the human loses.

The AI ship plays by the same rules it was trained on in `GameEnv`:
- each step the network sees the state at the end of the previous step, then the ship moves, shoots and uses missile/bomb before enemies, asteroids and bullets move;
- its bullets are absorbed by asteroids (with a spark), after the bullet-enemy check as for the human;
- its bombs use its own bomb level: the spawn freeze (lv8+) applies to both ships, the lv10 protective field protects only the ship that dropped the bomb (the human's field no longer protects the AI, and vice versa).

Aimed enemies target the nearest living ship at the moment they fire, so the human is also targeted less often when the AI is closer. Coop games are **not** recorded for imitation learning: the shared score includes the AI's kills and the allied ship does not exist in training.

---

## Gameplay Mechanics

### Player Controls

| Key | Action |
|-----|--------|
| `Arrow Keys` | Move (4 directions) |
| `SPACE` (hold) | Shoot |
| `M` | Fire missile (one per key press) |
| `B` | Detonate bomb (one per key press) |
| `ESC` | Return to main menu |

After game over: type a name (max 10 characters, `BACKSPACE` to delete), `ENTER` to save it (non-empty name) to the Top 10 leaderboard, then `ENTER` again to restart.

Each simulation step applies the human's actions in the same order as `GameEnv`: movement first, then shot, missile and bomb (`M`/`B` presses are collected once per rendered frame and applied at the next step), then the rest of the world moves.

- **Speed**: Base 6 px/frame (at 60 FPS), +10% per engine level up to 2x (12 px/frame); fractional speeds are kept with sub-pixel movement, so every engine level is effective
- **Health**: 3 lives + shield HP (0-5) from shield upgrades; the shield absorbs hits before lives
- **Invincibility**: 2.0–4.5 seconds after any hit (shield or life): 2 s base, +0.5 s every 2 shield levels

### Enemies

16 unique pixel-art enemy models (chosen at random) spawn from the right edge and scroll left at 3-6 px/frame (level 1). Four shooting behaviors unlock progressively:

| Type | Unlock Level | Base Fire Delay | Behavior |
|------|-------------|------------|----------|
| **Basic** | 3 | 2200 ms | Straight shot leftward |
| **Fast** | 4 | 1400 ms | Straight shot, high frequency |
| **Aimed** | 6 | 2800 ms | Calculates angle to the player (in Coop: the nearest living ship at each shot) |
| **Burst** | 10 | 3200 ms | 3-bullet spread (-12, 0, +12 degrees) |

Fire delays are divided by the difficulty factor (`1.12 ^ (level - 1)`) with a floor of 600 ms. Enemies only fire once fully on screen, and the first shot is staggered by an extra 0.5-1.5 s.

Shooter probability per enemy is a lookup table indexed by level (`ENEMY_SHOOTER_CHANCE`): 0% at levels 1-2, 5% at level 3, rising step by step to 62% at level 20 and staying there for higher levels. A shooter's type is picked uniformly among the unlocked types. Ramming an enemy destroys it (no points) and damages the player.

### Obstacles

**Asteroids** are procedurally generated irregular polygons (7-10 vertices). Radius ranges from 20-45 px at level 1, growing by 2 px per level up to 40-65 px (reached at level 11). Speed starts at 2-4 px/frame and grows by +0.3 / +0.4 px/frame per level. Asteroids are indestructible except by bombs (bomb level 2+): bullets are absorbed with a spark, and colliding with an asteroid damages the player without destroying it.

### Power-ups & Upgrades

Enemies destroyed by bullets or missiles have a **15% drop chance** for one of 4 power-up types (chosen uniformly). Kills by bomb or by ramming never drop power-ups. Power-ups drift left at 2 px/frame (clamped only vertically to the screen) and disappear when they leave the screen on the left, like missile pickups.

| Type | Color | Icon | Effect |
|------|-------|------|--------|
| **Engine** | Green | M | +10% speed per level (max 2x) |
| **Weapon** | Orange | W | +1 bullet on even levels (max 5); fire delay -15% from level 1, -30% at level 10; changes the player ship sprite |
| **Shield** | Blue | S | +1 max shield HP every 2 levels (max 5), refills the shield, +invincibility duration |
| **Bomb** | Purple | B | +1 bomb (capacity +1 every 2 levels, max 5), unlocks effects |

Each category has **10 upgrade levels**. Key milestones:

**Weapon progression:**
| Level | Bullets | Fire Delay |
|-------|---------|-----------|
| 0 | 1 | 250 ms |
| 1 | 1 | 212 ms (-15%) |
| 2 / 3 | 2 | 212 ms |
| 4 / 5 | 3 | 212 ms |
| 6 / 7 | 4 | 212 ms |
| 8 / 9 | 5 | 212 ms |
| 10 | 5 | 175 ms (-30%) |

The fire-delay reduction is persistent: leveling up never slows the fire rate.

**Bomb unlock chain:**
| Level | Effect |
|-------|--------|
| 1+ | Destroys all enemies (100 pts each) and all enemy bullets |
| 2+ | Also destroys asteroids |
| 6+ | Double score on bomb kills (200 pts) |
| 8+ | Freezes enemy and asteroid spawns for 1 second |
| 10 | Protective field for 2 seconds (all collisions of the ship that dropped the bomb ignored) |

### Weapons

- **Bullets**: 12 px/frame, animated 3-frame sprites with flame trail, 1-5 per volley depending on weapon level. Score: 100 pts/kill
- **Missiles**: Fly straight to the right at 14 px/frame and pierce through up to 5 enemies they touch (no homing). Score: 150 pts/kill, with explosion and drop chance, including the kills of the frame in which the missile is spent or leaves the screen. Ammo comes from floating missile pickups that spawn from the right edge every 8 s at level 1; the interval is multiplied by the difficulty factor and capped at 15 s (from level 7)
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
│  │   .txt)      │  │ 200 headless)│  │  gens, last 2)│   │
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

Both loops advance the simulation in fixed steps of `1000 / FPS` ms with `dt = 1.0`: each `GameEnv.step()` is one step, while `Game.run()` accumulates real time and runs one `_update(1.0)` per elapsed step (at most 5 per rendered frame; after a stall the excess time is dropped, so the game slows down instead of letting entities jump through each other). Entities keep a sub-pixel remainder and move their rect by `round()`, so fractional speeds (enemies, asteroids, angled enemy bullets, engine upgrades) are not truncated: effective speeds match the declared values and are identical in play and in training. Game timers (spawns, cooldowns, invincibility) still use `pygame.time.get_ticks()`: real time in `Game`, virtual time in `GameEnv`.

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
| `out[3]` | > 0.5 | Fire missile (held, at most one every 0.5 s) |
| `out[4]` | > 0.5 | Detonate bomb (rising edge) |

In `GameEnv._apply_actions` and for the Coop AI ship, a missile is fired whenever the decoded action is True and at least `AI_MISSILE_INTERVAL` (500 ms) have passed since the previous AI missile, so a network that keeps the output high uses its missiles as it collects them. The bomb fires only on the step in which the decoded action goes from False to True, like a human key press: holding the output high does not empty the stock on consecutive steps. `tanh` is the only `activation_options` value, so activation mutations can no longer turn an output into `sigmoid` ([0, 1]) or `relu` ([0, ∞)), which cannot go below -0.2 (no up/left); nodes of older genomes that already use them still load and run, and are replaced by `tanh` when their activation mutates.

### Fitness Function

Fitness is computed by `_compute_fitness(env)` in `src/neat_trainer.py` from the statistics counters of `GameEnv` at the end of each episode; a genome's fitness is the **mean over its episodes** (2 per generation, see Training Pipeline). Multi-component fitness with 16 terms (the game score itself is not a term). Time-based terms are in seconds of game time (`frames / FPS`), so the weights do not depend on `settings.FPS`:

```
fitness = enemies_killed × 50          # bullet kills
        + missile_kills × 50           # enemies killed by missiles
        + bomb_kills × 50              # enemies destroyed by bombs
        + accuracy_bonus               # (shots_hit / shots_fired) × enemies_killed × 10 (bullet kills only)
        + dodges × 5                   # close passes survived (see below)
        + seconds × 1.5                # survival time
        - lives_lost × 200             # death penalty
        + level² × 50                  # progression bonus
        + powerups_collected × 50
        + missiles_collected × 30
        + total_upgrade_levels × 20
        - shield_hits × 15
        - shield_breaks × 120
        - idle_seconds × 15            # time in which the ship did not move
        + near_misses × 3              # sprites that came within 60 px
        + max(0, approach_seconds) × 3 # net time spent moving towards the nearest power-up
        - shots_fired × 0.01
```

`FITNESS_VERSION` (currently 4) identifies this formula together with the `GameEnv` rules and the validation suites; it must be incremented whenever any of them changes (see Checkpointing & Persistence).

How the counters are measured in `GameEnv`:
- **Kills**: `enemies_killed` counts bullet kills, `missile_kills` the kills of every missile (counted on the snapshot taken before `missiles.update()`, so the kills of the frame in which a missile is spent or leaves the screen are included), `bomb_kills` the enemies cleared by a bomb.
- **Idle**: a frame is idle when the ship's rect did not change after the movement and the screen clamp, so holding a direction against an edge counts as idle.
- **Power-up approach**: a net counter: `+1` for a frame in which the ship's own movement brings it closer to the nearest power-up, `-1` for a frame in which it moves it away, measured against the power-up's position before it drifts (power-ups drifting towards a still ship do not count). Oscillating near a power-up therefore earns nothing; the fitness uses the balance clipped at 0.
- **Near-miss**: an enemy, asteroid or enemy bullet counts once, the first time its center comes within 60 px of the ship's center (the mark is stored on the sprite).
- **Dodges**: an enemy or asteroid that leaves the screen after a near-miss without having hit the ship; enemies destroyed by any weapon or by ramming never count, nor do asteroids that hit the ship.

Fitness is not clipped at 0. `shots_fired` counts volleys, `shots_hit` counts bullets that hit at least one enemy. A genome whose evaluation raises an exception gets a fitness of -10000 (with no imitation bonus) and the traceback is printed to the console. The imitation bonus (below) is added on top.

**Per-step reward** (`GameEnv.step()`): `step()` also returns a shaped reward, which `NeatTrainer` does not use (fitness comes only from `_compute_fitness`):
- Survival bonus: `+0.02` per frame
- Kill: `+5` per enemy (bullet, missile or bomb)
- Death penalty: `-200` per life lost
- Shot penalty: `-0.01` per volley fired
- Shield: `-15` per shield hit, `-60` per shield break
- Pickups: `+10` per upgrade power-up, `+5` per missile pickup, `+5` per upgrade level gained
- Level up: `+level² × 5`
- Near-miss bonus: `+3.0` the first time an enemy, asteroid or enemy bullet comes within 60 px of the player
- Power-up approach: `+0.1` per frame in which the ship moves closer to the nearest power-up, `-0.1` per frame in which it moves away

**Early termination** (constants in seconds converted with `FPS`, `EARLY_STOP_*`, shared by headless workers and visual mode through `_early_stop()`): from 6 s (`EARLY_STOP_MIN_FRAME`, 360 frames) the episode is checked every 5 s (`EARLY_STOP_INTERVAL`, 300 frames), i.e. at frame 600, 900, 1200, … The genome is stopped if, from generation 10 (`EARLY_STOP_LIVES_GEN`), it has lost 2 of its 3 lives (`EARLY_STOP_MAX_LIVES_LOST`), or if it has been idle (not moving, edges included) for at least 70% of the frames (`EARLY_STOP_IDLE_RATIO`). Validation episodes have no early termination.

### Imitation Learning

Human gameplay is recorded during **Play** mode only via the `HumanRecorder` class (one frame per simulation step). Coop games are not recorded; sessions recorded in Coop by earlier versions (indistinguishable, no flag) are still in `ai/human_sessions/` until rotation removes them.

**Recording format** (per frame):
```python
(observation_52dim, action_dict)  # action = {vertical, horizontal, shoot, missile, bomb}
```

Each pair is the same one the network sees in `GameEnv`: the observation is the state at the end of the previous step (`Game._pending_obs`, captured at the end of `_update`, at game start and at restart), the action is what the human does in this step. With opposite keys held, the recorded direction is the one the ship actually moves in (down and right win, as in `Player.update`).

**Sessions stored**: `ai/human_sessions/session_{timestamp}.pkl`, a dict `{frames, score, num_frames, timestamp, version}`, written at game over (max 20 retained). Sessions without `version` (version 1, recorded before October 2026) pair the action with the observation taken after that step's shot, missile and bomb, and with opposite keys held record the direction opposite to the movement; they are still used.

**Bonus application** (from generation 10, `IMITATION_START_GEN`, computed for every genome: in headless mode by the worker process right after the episodes, in visual mode in the main process):
1. Load the human sessions once per training run, before the worker pool is created, and use the 5 valid sessions with the highest score (`IMITATION_TOP_SESSIONS`; ties go to the most recent), each weighted by `min(4, ln(1 + score / 1000))`, so that no single game dominates; unreadable or malformed sessions are skipped with a warning
2. Keep 200 evenly spaced frames per session (`IMITATION_SAMPLES_PER_SESSION`; fewer if the session is shorter); only these are kept in memory and passed to the workers
3. Feed each observation to the evolved network
4. Compare the network's decoded action with the recorded human action on movement and shooting only: movement axes 1 (match) / 0.5 (off by one) / 0 (opposite), shoot 1 if equal; average → 0-1. Missile and bomb are excluded: human presses are so rare that comparing them would only reward never using them
5. Subtract the similarity expected by chance: `sim_chance` is the similarity that a policy with the genome's own action frequencies would get if it ignored the observation (weighted Cohen's kappa, computed from the genome's and the humans' weighted action frequencies)
6. Apply bonus: `max(0, sim - sim_chance) / (1 - sim_chance) × 300` (max 300 fitness points)

Any constant policy (standing still, always shooting, holding a direction) and any random policy independent of the state get a bonus of 0: only acting like the human in the same situations is rewarded. With the 15 sessions in the repository and the checkpoint-80 population, the bonus is between 0 and ~11 (the evolved genomes show almost no state-dependent agreement with the human player yet); a baseline equal to the best constant policy would have given 0 to every genome, a baseline equal to "do nothing" would have given 188 to "always shoot".

This acts as a **soft behavioral prior** — once the bonus is active, genomes that play like humans receive a fitness bonus, steering evolution away from degenerate strategies.

### Training Pipeline

```
1. Load NEAT config + latest readable checkpoint (if any)
   - pop_size is set first: 200 in headless (HEADLESS_POP_SIZE), the config value (150) in visual
   - a resumed run uses ai/neat_config.txt (restore_checkpoint(new_config=...)), keeping the
     compatibility threshold adapted in the checkpoint; the restored species set is re-bound to
     the live species_set_config so threshold changes take effect
   - if the checkpoint was saved with a different FITNESS_VERSION, the species fitness
     history is reset (it would otherwise be on the old scale and mark every species stagnant)
   - the validation record of ai/best_genome.pkl is read from ai/best_genome_meta.json, or
     recomputed with the confirmation suite if the sidecar is missing or stale
2. For each generation:
   a. Evaluate all genomes on the same 4 seeds (common random numbers: generation × 1000 + k,
      new each generation); fitness = mean of the 4 episodes (EPISODES_PER_GENOME)
      - headless: persistent multiprocessing.Pool, imap_unordered (chunksize=1), one task per genome
      - visual: serial, in-process, rendering 1 frame out of 4
      - Each evaluation: compile the genome's FeedForwardNetwork into a generated Python function
        (same outputs as neat's activate(), about 2x faster) → run the GameEnv episodes
      - Episode length cap: 30 s → 180 s of game time (1800 → 10800 frames at FPS = 60),
        linear over the first 50 generations
      - Virtual time system ensures deterministic evaluation
        (pygame.time.get_ticks patched to frame_count × 1000 / FPS, ≈16.7 ms per frame)
   b. Apply imitation learning bonus (from gen 10; headless: computed in the workers)
   c. Validation in two stages, full episodes (10800 frames, no early termination, no imitation):
      - selection: the 5 genomes with the highest game fitness (without imitation) play the
        10 fixed VALIDATION_SEEDS
      - confirmation: if the best selection mean beats the record, that genome plays the 40
        fixed CONFIRMATION_SEEDS and replaces ai/best_genome.pkl if its confirmation mean beats
        the record (the record is the confirmation mean of best_genome.pkl)
      (headless: in the Pool; genomes already played in the session, e.g. elites, are cached)
   d. Adjust the compatibility threshold towards 5-10 species (Anti-Stagnation)
   e. Append the generation to ai/training_history.jsonl and refresh the dashboard data
   f. NEAT reproduction + speciation (neat-python)
   g. Checkpoint every 10 generations (only the last 2 kept), and on interruption
```

**Parallelization**: The `Pool` is created once per training run. Each worker process initializes with `SDL_VIDEODRIVER=dummy`, `SDL_AUDIODRIVER=dummy` and a 1×1 display for headless Pygame operation, receives the imitation samples and the human action frequencies through the pool initializer, and lowers its priority to `BELOW_NORMAL` on Windows. Genome evaluation and validation episodes are distributed via `pool.imap_unordered()`; the summary screen is redrawn (and `ESC`/`D` polled) every 5 completed genomes and after every validation episode.

**Cost** (8-core machine, 7 workers, measured on the checkpoint-80 population): about 150 worker-µs per game frame; a generation of 200 genomes × 2 episodes took ~13 s plus ~5 s of validation while genomes die early (~1500 frames per episode), and would take ~95 s plus up to ~12 s of validation if every episode reached the 180 s cap. With 4 episodes per genome the evaluation time doubles; a confirmation (40 full episodes) adds roughly 8 s to the generations that need one.

### Training Dashboard

`dashboard/index.html` is a self-contained page (vanilla JavaScript and SVG charts, no external libraries, works offline from `file://`). Training opens it in the default browser (`DASHBOARD_AUTO_OPEN`) and `D` in the training window reopens it. It reads two scripts that `src/training_dashboard.py` writes next to it, atomically, loaded with `<script>` tags because browsers block `fetch` on `file://`:
- `dashboard/data.js` (`window.YS_DATA`), rewritten at the end of every generation: the whole history, the detail of the last generation (top 10 genomes, species table, record with per-seed fitness and fitness breakdown) and the training parameters, including the fitness components with their weights and descriptions (`FITNESS_COMPONENTS` in `neat_trainer.py`, kept next to `_compute_fitness`);
- `dashboard/live.js` (`window.YS_LIVE`), rewritten at most once per second: phase (`avvio`, `rivalutazione`, `valutazione`, `validazione`, `fermo`), progress, generation and the stamp of `data.js`.

The page polls `live.js` every 3 s and reloads `data.js` only when its stamp changes. The history is kept in `ai/training_history.jsonl` (one JSON line per generation, appended, committed like the other training state), so the dashboard also shows previous sessions; if a generation appears twice (resume from an older checkpoint) the last line wins. Each line holds: generation, fitness version, session, duration, episode cap, population fitness summary (mean, median, 10th/90th percentile, min, max, sd), game statistics averaged over the population and over the top 10% by fitness (score, survival seconds, level, lives lost, kills per weapon, missiles collected/fired, bombs used, power-ups, idle %, accuracy), mean fitness components, the generation's best genome, the best of the selection stage (mean, sd and game statistics on the validation seeds), the confirmation played in the generation if any (same fields, on the confirmation seeds), the record and the record's game statistics, species (count, threshold, max stagnation, size per species) and network size (mean/max nodes and enabled connections).

Sections: status bar (live phase and progress), a plain-language verdict (record improvements, population trend, how the record plays, behavior warnings, species, speed), six indicator cards with sparklines, progress (record, best of the selection stage and confirmations, population fitness with rolling mean and 10–90% band, the record's per-seed confirmation fitness), gameplay (score, survival, level, kills per weapon), behavior (idle time, accuracy, lives lost, missiles, bombs, power-ups), fitness breakdown (record vs population mean, and the main components over time), species (stacked sizes and table with stagnation), networks and timing, the last generation's top 10, and a glossary with the training parameters. Trend badges compare the mean of the last 15 generations with the previous 15 and report "stable" when the difference is within the generation-to-generation noise (|z| < 2); the verdict and the trends use only the current `FITNESS_VERSION`, and charts mark version changes and resumed sessions with dashed lines. The range selector (last 50, last 200, all generations) and the theme (auto, light, dark) are stored in the browser's `localStorage`.

### NEAT Hyperparameters

From `ai/neat_config.txt`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Population size | 150 | Forced to 200 in headless training (`HEADLESS_POP_SIZE`) |
| Fitness criterion | max | `fitness_threshold = 99999`, `no_fitness_termination = True`, `reset_on_extinction = True` |
| Initial topology | 52→5 fully connected (`full_direct`) | No hidden nodes, feed-forward |
| Activation | tanh | Only option (`activation_options = tanh`): sigmoid/relu nodes of older genomes mutate back to tanh |
| Aggregation | sum | Not mutated |
| Weight mutation rate | 40% | Power 0.5, replace rate 10%, range ±5 |
| Bias mutation rate | 70% | Power 0.5, replace rate 10%, range ±5 |
| Connection add prob | 40% | Structural growth |
| Node add prob | 35% | Hidden neuron insertion |
| Connection delete prob | 20% | Pruning |
| Node delete prob | 10% | Simplification |
| Compatibility threshold | 3.0 | Initial species boundary for a new run (adjusted at runtime towards 5-10 species; a resumed run keeps the adapted value) |
| Compatibility coefficients | disjoint 1.0, weight 0.5 | |
| Stagnation limit | 25 gens | Per-species patience (`species_fitness_func = mean`) |
| Species elitism | 3 | Species protected from stagnation removal |
| Elitism | 3 | Top genomes preserved per species |
| Survival threshold | 20% | Only top 20% reproduce |
| Min species size | 3 | Prevents tiny species |

### Anti-Stagnation Mechanisms

1. **Speciation control** (`_check_and_fix_stagnation()`, after every generation): the compatibility threshold keeps the number of species between `SPECIES_TARGET_MIN` (5) and `SPECIES_TARGET_MAX` (10): below it is multiplied by 0.95 (min 1.0), above by 1.05 (max 6.0). The change applies to the next speciation, also after a resume. With the former limits (only ≤ 2 or > 20 species) the population settled on ~20 species of ~10 genomes and no species lived more than ~40 generations
2. **Species stagnation** (neat-python `DefaultStagnation`): species whose mean fitness does not improve on their best for 25 generations are removed (the 3 best species by current mean fitness are always protected)
3. **Noise reduction**: common seeds for the whole generation and a 4-episode mean make species fitness and elites less dependent on a lucky episode (with 2 episodes, only 6 of the top 20 genomes on one pair of seeds were in the top 20 on another pair)

The former injection of fresh/mutated genomes on fitness plateaus was removed: it ran inside the evaluation callback, left genomes without fitness (neat-python's `Population.run()` raises `RuntimeError`) and the injected genomes never reached reproduction.

### Checkpointing & Persistence

| File | Format | Content |
|------|--------|---------|
| `ai/neat-checkpoint-{N}` | gzip-compressed pickle | Full NEAT state: generation, config (with `fitness_version`), population, species, random state |
| `ai/best_genome.pkl` | Pickle | The genome with the best confirmation mean (used for Coop mode) |
| `ai/best_genome_meta.json` | JSON | Validation record of `best_genome.pkl` on the confirmation suite: `validation_mean`, per-seed fitness, mean statistics (`validation_stats`), `validation_seeds` (the confirmation seeds), `max_frames`, `fitness_version`, `genome_key`, generation, timestamp |
| `ai/training_history.jsonl` | JSON Lines | One line per evaluated generation, read by the training dashboard (see Training Dashboard) |
| `ai/human_sessions/session_*.pkl` | Pickle | Recorded human (observation, action) frame pairs + final score + format version |
| `scores.json` | JSON | Top-10 human player leaderboard (`name`, `score`), UTF-8, written atomically (temporary file + `os.replace`). A missing, empty or invalid file reads as an empty leaderboard (malformed entries are skipped); an unparsable file is renamed to `scores.json.bak` before the next save; if the file cannot be written the score is not saved and the game goes on |

Checkpoints are saved every 10 generations by `_RollingCheckpointer` (the first one right after the first generation of a resumed run), and also when training is interrupted with `ESC`, by closing the window or with Ctrl+C, if at least one generation was completed in the session. Each checkpoint is written to a temporary file (`neat-checkpoint-tmp`) and renamed, so an interrupted write never leaves a truncated checkpoint; the rotation keeps the 2 most recent up to the one just saved (`MAX_CHECKPOINTS = 2`). On startup the readable checkpoint with the highest generation is resumed: unreadable files are skipped with a warning and non-numeric suffixes (e.g. `.bak`) are ignored.

`ai/best_genome.pkl` (same format as before: a pickled `DefaultGenome`) and its sidecar are written atomically whenever a confirmed genome beats the record. If the sidecar is missing, refers to another genome, or was written with another `FITNESS_VERSION`, confirmation seeds or episode cap, the existing `best_genome.pkl` is re-validated at startup to set the record. These files are committed to the repository so training can continue across machines.

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
├── dashboard/
│   └── index.html           # Training dashboard (data.js and live.js are generated)
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
│   ├── neat_trainer.py      # NeatTrainer — evolution loop, multiprocessing, summary screen
│   ├── training_dashboard.py  # History and live state for the HTML dashboard
│   └── human_recorder.py    # Records human gameplay sessions for imitation learning
│
├── ai/
│   ├── neat_config.txt      # NEAT hyperparameter configuration
│   ├── best_genome.pkl      # Best validated genome (for Coop mode)
│   ├── best_genome_meta.json  # Validation record of best_genome.pkl (created by training)
│   ├── training_history.jsonl # Per-generation training history (created by training)
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

All game constants are defined in [settings.py](settings.py) (NEAT parameters are in `ai/neat_config.txt`; training constants such as `MAX_EPISODE_SECONDS`, `EARLY_STOP_*`, `HEADLESS_POP_SIZE`, `EPISODES_PER_GENOME`, `VALIDATION_*`, `IMITATION_*` and `FITNESS_VERSION` are at the top of `src/neat_trainer.py`). Speeds are in px per frame at 60 FPS:

### Display

| Constant | Value | Description |
|----------|-------|-------------|
| `SCREEN_WIDTH` | 1280 | Window width in pixels |
| `SCREEN_HEIGHT` | 740 | Window height in pixels |
| `FPS` | 60 | Target frame rate and simulation step (`1000 / FPS` ms) of both the game and the training virtual clock |

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
| `POWERUP_SPEED` | 2 | Drift speed (px/frame); the power-up disappears past the left edge |
| `BOMB_FREEZE_DURATION` | 1000 | Spawn freeze on bomb (ms) |
| `BOMB_FIELD_DURATION` | 2000 | Protective field on lv10 bomb (ms) |

---

## Technical Details

### Procedural Audio Synthesis

The `SoundManager` generates **all audio at runtime** using waveform math (44.1 kHz, 16 mixer channels) — no external audio files are needed. Audio is used only in Play and Coop modes, not during training. If the mixer cannot be initialized (no audio device), the game runs muted.

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

**Background music**: ~17.5-second loop (8 bars at 110 BPM) in Am (Am-F-C-G, played twice), looping on a dedicated channel, paused at game over and stopped when the game returns to the menu. Composed of 5 layers: pulsing bass, chord pad, arpeggio, hi-hat, and kick drum — all synthesized from sine and square waveforms.

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
