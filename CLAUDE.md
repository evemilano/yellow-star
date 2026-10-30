# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Progetto

Yellow Star: shooter a scorrimento orizzontale in **pygame-ce** con pipeline **NEAT** (`neat-python`) che evolve reti neurali per giocare. Modalità dal menu: gioco umano (registrato per imitation learning), training headless multiprocesso, training visuale, coop umano+AI. Commenti, docstring e testi UI sono in italiano.

Il `README.md` (in inglese) documenta in dettaglio meccaniche, fitness, reward, hyperparametri NEAT e biomi, ed è allineato al codice di ottobre 2026. Descrive il comportamento attuale anche dove è difettoso: se modifichi una meccanica, una soglia o una formula, aggiorna la sezione corrispondente. In caso di conflitto vale il codice.

## Comandi

Venv in `venv\` (Python 3.14, pygame-ce 2.5.6). Da lanciare sempre dalla root del repo: `settings.py` è importato come modulo top-level (`from settings import ...`).

```bash
run.bat                                   # attiva il venv e avvia main.py
venv/Scripts/python.exe main.py           # equivalente

# Niente test suite né linter. Verifica minima:
venv/Scripts/python.exe -m py_compile main.py settings.py generate_enemies.py src/*.py

# Smoke test headless di GameEnv (stessa init dei worker di training)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy venv/Scripts/python.exe -c "
import pygame; pygame.init(); pygame.display.set_mode((1,1))
from src.game_env import GameEnv
env = GameEnv(); obs = env.reset(seed=1)
for _ in range(600):
    obs, r, done = env.step({'vertical':0,'horizontal':0,'shoot':True,'missile':False,'bomb':False})
    if done: break
print(len(obs), env.frame_count, env.score, env.lives_lost)"

venv/Scripts/python.exe generate_enemies.py   # rigenera (sovrascrive) assets/images/enemies/*.png
```

Il training si avvia solo dal menu (non c'è CLI); ESC o chiusura finestra lo interrompono (`_TrainingInterrupted`).

## Architettura

`main.py` → `MainMenu` → `Game` (play / coop) oppure `NeatTrainer.train(headless=...)`. Gli import dei moduli pesanti sono lazy dentro `main()`; il guard `if __name__ == "__main__"` è necessario per il `multiprocessing.Pool` su Windows (spawn).

**Due implementazioni parallele del game loop** — punto critico:
- `src/game.py` `Game._update()`: gioco reale, tempo da `pygame.time.get_ticks()`, `dt` normalizzato su `FPS`, audio, HUD, leaderboard, coop AI.
- `src/game_env.py` `GameEnv._update()`: replica *a mano* la stessa logica per il training (`reset(seed)` / `step(actions) -> (obs, reward, done)`), senza audio/HUD, rendering opzionale.

Ogni modifica alle regole di gioco (spawn, collisioni, power-up, bombe, livelli) va riportata in **entrambi** i file, altrimenti l'AI si allena su un gioco diverso da quello giocato. Le collisioni proiettili×bersagli usano `groupcollide_fast` (in `game_env.py`, stessa semantica di `pygame.sprite.groupcollide`), condivisa dai due loop. Le entità (`player.py`, `enemy.py`, `asteroid.py`, `missile.py`, `powerup.py`, `upgrades.py`, `level.py`...) sono condivise.

**Tempo virtuale deterministico**: `GameEnv.step()` fa monkey-patch di `pygame.time.get_ticks` con `frame_count * (1000/FPS)` per la durata del frame e lo ripristina nel `finally`. Le entità leggono il tempo solo via `pygame.time.get_ticks()` (cooldown di sparo, invincibilità, fire delay nemici): nel codice delle entità non usare altre sorgenti di tempo. Il seed per genoma è `genome.key + generation * 10000`.

**Contratto osservazione/azione** (52 input, 5 output):
- `build_observation()` in `game_env.py` è l'unica fonte del vettore a 52 valori, usata da `GameEnv`, da `HumanRecorder` e dalla nave AI in coop. Il layout è documentato nella sua docstring.
- La decodifica dell'output (soglie ±0.2 sul movimento, 0.3 sparo, 0.5 missile/bomba) è duplicata in `neat_trainer._decode_actions_static` e `Game._decode_ai_actions`: tenerle allineate.
- Cambiare dimensione o ordine dell'osservazione richiede di aggiornare `num_inputs` in `ai/neat_config.txt` e rende inutilizzabili checkpoint, `best_genome.pkl` e sessioni umane registrate.

**Training** (`src/neat_trainer.py`): headless usa un `Pool` persistente (`cpu_count-1` worker, `_init_worker` imposta `SDL_VIDEODRIVER`/`SDL_AUDIODRIVER=dummy`, priorità bassa su Windows e riceve i campioni dell'imitation bonus) con `imap_unordered` su `_eval_genome_worker` (top-level per essere picklable); il visuale valuta in-process con `_eval_single`. Le reti si valutano con `compile_network` (funzione Python generata, output identici bit per bit a `FeedForwardNetwork.activate`): la funzione compilata non è picklable, non salvarla sul genoma. La fitness è in `_compute_fitness(env)` e somma i contatori statistici di `GameEnv` (il `reward` restituito da `step()` è un segnale separato). Poi: early stop dei genomi scarsi, bonus imitation dopo `IMITATION_START_GEN` (campioni delle ultime 5 sessioni di `ai/human_sessions/`, caricati in `train()` prima del `Pool`; in headless il bonus lo calcola il worker, nel visuale `_collect_stats`), gestione stagnazione (`_check_and_fix_stagnation` / `_inject_fresh_genomes`), dashboard pygame (`_draw_section_*`). Le costanti in cima al file (`MAX_FRAMES_PER_EPISODE`, `EARLY_STOP_*`) sono in frame e i commenti presumono 30 FPS, mentre `settings.FPS` è 60.

**Sfondo** (`src/parallax.py`): `ParallaxBackground` ruota biomi procedurali (sottoclassi di `Biome`: Mountain, Sea, City, Jungle, Desert, Arctic) con crossfade ogni `BIOME_DURATION` ms e passa a `SpaceBiome` da `BIOME_SPACE_LEVEL`. Il bioma successivo è costruito in un thread `BIOME_PREBUILD_LEAD` ms prima dello switch: le `build()` devono usare solo il proprio `random.Random` e non chiamare `convert()`. In `GameEnv` viene creato solo con `render=True`.

**Asset procedurali**: audio (`sounds.py`, nessun file audio), esplosioni, asteroidi e sprite nemici (`generate_enemies.py` → PNG). Gli sprite della nave giocatore (`assets/images/player/ship_01..10.png`, uno per livello weapon) sono caricati da `ship_sprites.py`.

## Stato persistente versionato

`ai/neat-checkpoint-*`, `ai/best_genome.pkl`, `ai/human_sessions/*.pkl` e `scores.json` sono **committati di proposito** (training portato tra PC). Giocare o allenare li modifica/crea/cancella (rotazione a 2 checkpoint e a 20 sessioni): non ripulirli né rigenerarli senza richiesta. La modalità coop richiede `ai/best_genome.pkl`. Il training riprende automaticamente dal checkpoint con generazione più alta.
