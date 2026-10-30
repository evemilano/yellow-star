# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Progetto

Yellow Star: shooter a scorrimento orizzontale in **pygame-ce** con pipeline **NEAT** (`neat-python`) che evolve reti neurali per giocare. Modalità dal menu: gioco umano (registrato per imitation learning), training headless multiprocesso, training visuale, coop umano+AI (non registrata). Commenti, docstring e testi UI sono in italiano.

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

Il training si avvia solo dal menu (non c'è CLI); ESC o chiusura finestra lo interrompono (`_TrainingInterrupted`, con checkpoint di interruzione). Per provare `train()` da script: guard `__main__` (Pool spawn) e monkeypatch di tutti i path del modulo (`AI_DIR`, `BEST_GENOME_PATH`, `BEST_GENOME_META_PATH`, `CHECKPOINT_PREFIX`, `SESSIONS_DIR`) verso una copia, altrimenti scrive in `ai/`; `MAX_FRAMES_PER_EPISODE`/`MIN_FRAMES_PER_EPISODE` patchati nel processo principale arrivano ai worker negli argomenti dei task, le altre costanti no.

## Architettura

`main.py` → `MainMenu` → `Game` (play / coop) oppure `NeatTrainer.train(headless=...)`. Gli import dei moduli pesanti sono lazy dentro `main()`; il guard `if __name__ == "__main__"` è necessario per il `multiprocessing.Pool` su Windows (spawn).

**Due implementazioni parallele del game loop** — punto critico:
- `src/game.py` `Game._update()`: gioco reale, timer da `pygame.time.get_ticks()` reale, audio, HUD, leaderboard, coop AI. `Game.run()` è a passo fisso come GameEnv: accumula il tempo di `clock.tick` e chiama `_update(1.0)` ogni `1000/FPS` ms (`_sim_steps`, max `MAX_SIM_STEPS = 5` passi per frame, l'eccesso si scarta), poi un solo `_draw`; esce sempre con `stop_music()`. `_handle_events` imposta solo i flag keydown di missile/bomba: sparo (SPACE tenuto), missile e bomba dell'umano si applicano in `_update` dopo `player.update`, nell'ordine di `_apply_actions`.
- `src/game_env.py` `GameEnv._update()`: replica *a mano* la stessa logica per il training (`reset(seed)` / `step(actions) -> (obs, reward, done)`), senza audio/HUD, rendering opzionale.

Ogni modifica alle regole di gioco (spawn, collisioni, power-up, bombe, livelli) va riportata in **entrambi** i file, altrimenti l'AI si allena su un gioco diverso da quello giocato. Vale anche per la nave AI in coop, che gioca con le regole di GameEnv: `_update_ai_actions` a inizio `_update` (la rete vede lo stato di fine passo precedente, poi movimento, sparo, missile, bomba, come `_apply_actions`), `_update_ai` a fine passo (proiettili AI contro nemici e poi asteroidi, power-up, collisioni). `Game._use_bomb(player)` è unica per umano e AI: freeze degli spawn globale, campo lv10 per nave (`bomb_field_until` umano, `ai_bomb_field_until` AI). I nemici 'aimed' in coop ricevono `target_fn=_aimed_target` (nave viva più vicina, scelta a ogni sparo); single player e GameEnv passano solo `player`. Le collisioni proiettili×bersagli usano `groupcollide_fast` (in `game_env.py`, stessa semantica di `pygame.sprite.groupcollide`), condivisa dai due loop. Le entità (`player.py`, `enemy.py`, `asteroid.py`, `missile.py`, `powerup.py`, `upgrades.py`, `level.py`...) sono condivise: per esempio `PowerUp.update` fa solo il clamp verticale, quindi il power-up esce a sinistra e sparisce (come `MissilePowerUp`) in entrambi i loop.

**Movimento sub-pixel**: le entità non troncano `speed*dt`: tengono il resto frazionario in `_sub_x`/`_sub_y` (|resto| ≤ 0.5) e spostano il rect di `round()`. Il rect resta la posizione di riferimento, quindi clamp, cambio sprite e riposizionamenti esterni non richiedono risincronizzazioni. Il movimento del player è solo in `Player.move()` (umano, `GameEnv._apply_actions`, AI coop); per reset/restart usare `Player.reset_position()`, che azzera il resto. I punti dei missili si contano su uno snapshot del gruppo preso prima di `missiles.update()` (Game, `_update_ai`, GameEnv): un missile esaurito o fuori schermo esce dal gruppo dentro `update()`.

**Tempo virtuale deterministico**: `GameEnv.step()` fa monkey-patch di `pygame.time.get_ticks` con `frame_count * (1000/FPS)` per la durata del frame e lo ripristina nel `finally`. Le entità leggono il tempo solo via `pygame.time.get_ticks()` (cooldown di sparo, invincibilità, fire delay nemici): nel codice delle entità non usare altre sorgenti di tempo. `env.reset(seed)` risemina il `random` globale: in training ogni generazione usa gli stessi `EPISODES_PER_GENOME` (2) seed per tutti i genomi (`_generation_seeds`: `generation * 1000 + k`), la validazione i 10 `VALIDATION_SEEDS` fissi.

**Contratto osservazione/azione** (52 input, 5 output):
- `build_observation()` in `game_env.py` è l'unica fonte del vettore a 52 valori, usata da `GameEnv`, da `HumanRecorder` e dalla nave AI in coop. Il layout è documentato nella sua docstring.
- La decodifica dell'output (soglie ±0.2 sul movimento, 0.3 sparo, 0.5 missile/bomba) è duplicata in `neat_trainer._decode_actions_static` e `Game._decode_ai_actions`: tenerle allineate. Missile e bomba scattano sul fronte di salita (come il KEYDOWN umano) in `GameEnv._apply_actions` (`_prev_missile`/`_prev_bomb`, azzerati in `reset`) e in `Game._update_ai_actions` (`_ai_prev_*`, azzerati in `_restart`). `activation_options = tanh`: niente output sigmoid/relu, che non scendono sotto -0.2.
- `HumanRecorder` registra la coppia di GameEnv: `Game._pending_obs` (obs di fine passo precedente, presa a fine `_update`, in `__init__` e in `_restart` da `_capture_pending_obs`) con le azioni del passo; con tasti opposti vince la direzione di `Player.update` (giù, destra). Le sessioni nuove hanno `'version': SESSION_VERSION` (2); quelle senza campo sono della versione 1 (obs presa dopo sparo/missile/bomba, direzione invertita con tasti opposti) e restano in uso.
- Cambiare dimensione o ordine dell'osservazione richiede di aggiornare `num_inputs` in `ai/neat_config.txt` e rende inutilizzabili checkpoint, `best_genome.pkl` e sessioni umane registrate.

**Training** (`src/neat_trainer.py`): headless usa un `Pool` persistente (`cpu_count-1` worker, `_init_worker` imposta `SDL_VIDEODRIVER`/`SDL_AUDIODRIVER=dummy`, priorità bassa su Windows e riceve i campioni dell'imitation bonus) con `imap_unordered` su `_eval_genome_worker` (top-level per essere picklable); il visuale valuta in-process con `_eval_single`. Le reti si valutano con `compile_network` (funzione Python generata, output identici bit per bit a `FeedForwardNetwork.activate`): la funzione compilata non è picklable, non salvarla sul genoma. La fitness è in `_compute_fitness(env)`, somma dei componenti `comp_*` calcolati dai contatori statistici di `GameEnv` (il `reward` restituito da `step()` è un segnale separato); i termini a tempo sono in secondi (`frame / FPS`) e le durate in cima al file sono in secondi convertite con `FPS` (`MAX_EPISODE_SECONDS = 180`, `MIN_EPISODE_SECONDS = 30`, `EARLY_STOP_*`). Contatori di `GameEnv` che la fitness assume: uccisioni separate per arma (`enemies_killed` a proiettile, `missile_kills` sullo snapshot dei missili, `bomb_kills`), `idle_frames` sullo spostamento reale del rect (premere contro il bordo è idle), `powerup_approach_frames` solo per lo spostamento della nave, near-miss marcati sullo sprite (`_near_miss`), schivate = nemici/asteroidi usciti dopo un near-miss senza colpire (`_hit_player` sugli asteroidi). Ogni genoma gioca i seed della generazione e prende la media (`_eval_genome_worker` / `_eval_single`, early stop unico in `_early_stop`: idle ≥ 70% o, dalla gen 10, 2 vite perse). Bonus imitation dopo `IMITATION_START_GEN`: solo movimento e sparo, 200 frame equidistanti dalle 5 sessioni con lo score più alto, peso `min(4, ln(1 + score/1000))`, bonus = `max(0, sim - sim_caso) / (1 - sim_caso) * 300` con `sim_caso` dalle frequenze d'azione del genoma (kappa di Cohen: ogni policy costante prende 0); campioni e frequenze umane caricati in `train()` prima del `Pool`, in headless il bonus lo calcola il worker, nel visuale `_collect_stats`. A fine generazione `_validate_top_genomes` rigioca i 5 genomi con la fitness di gioco più alta su `VALIDATION_SEEDS` (episodi interi, niente early stop né imitation; nel Pool in headless, cache per `genome.key`) e salva `best_genome.pkl` + `best_genome_meta.json` se la media batte il record. Il restore usa `new_config=self.config` (pop_size e parametri da `neat_config.txt`, headless 200) tenendo il threshold adattato del checkpoint, e ricollega `p.species.species_set_config` alla config; se il checkpoint ha un'altra `fitness_version` (attributo della config pickled) azzera lo storico delle specie. Anti-stagnazione: solo threshold adattivo (`_check_and_fix_stagnation`) e `DefaultStagnation` (`species_fitness_func = mean`); non toccare `Population.population` dalla fitness function (neat 1.1.0 solleva `RuntimeError` per genomi senza fitness). Se cambiano `_compute_fitness`, le regole di GameEnv o la suite di validazione, incrementare `FITNESS_VERSION`. Dashboard pygame in `_draw_section_*`.

**Sfondo** (`src/parallax.py`): `ParallaxBackground` ruota biomi procedurali (sottoclassi di `Biome`: Mountain, Sea, City, Jungle, Desert, Arctic) con crossfade ogni `BIOME_DURATION` ms e passa a `SpaceBiome` da `BIOME_SPACE_LEVEL`. Il bioma successivo è costruito in un thread `BIOME_PREBUILD_LEAD` ms prima dello switch: le `build()` devono usare solo il proprio `random.Random` e non chiamare `convert()`. In `GameEnv` viene creato solo con `render=True`.

**Asset procedurali**: audio (`sounds.py`, nessun file audio), esplosioni, asteroidi e sprite nemici (`generate_enemies.py` → PNG). Gli sprite della nave giocatore (`assets/images/player/ship_01..10.png`, uno per livello weapon) sono caricati da `ship_sprites.py`.

## Stato persistente versionato

`ai/neat-checkpoint-*`, `ai/best_genome.pkl`, `ai/human_sessions/*.pkl` e `scores.json` sono **committati di proposito** (training portato tra PC); il training crea anche `ai/best_genome_meta.json` (record di validazione del best: media, seed, `fitness_version`, `genome_key`). Giocare o allenare li modifica/crea/cancella (rotazione a 2 checkpoint e a 20 sessioni): non ripulirli né rigenerarli senza richiesta. Checkpoint, `best_genome.pkl` e sidecar si scrivono in modo atomico (temporaneo + `os.replace`; il temporaneo dei checkpoint è `neat-checkpoint-tmp`); il checkpoint si salva anche all'interruzione se nella sessione è finita almeno una generazione; alla ripresa i checkpoint illeggibili e i suffissi non numerici vengono saltati. Se il sidecar manca o non corrisponde (genoma, versione, seed, frame), all'avvio `best_genome.pkl` viene rivalutato con la suite e il sidecar riscritto. `scores.py` scrive `scores.json` in modo atomico (`.tmp` + `os.replace`); se il file non è leggibile come JSON lo rinomina in `scores.json.bak` prima di salvare. La modalità coop richiede `ai/best_genome.pkl` e non registra sessioni (`main.py` la avvia senza `record_for_ai`); se il genoma non è caricabile o non è compatibile, `_load_ai_genome` ritorna False e la partita parte senza nave AI con un avviso a schermo. Il training riprende automaticamente dal checkpoint leggibile con generazione più alta.
