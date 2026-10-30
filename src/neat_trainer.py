"""Orchestrazione training NEAT per Yellow Star.

Gestisce:
- Evoluzione della popolazione NEAT
- Valutazione fitness dei genomi (con rendering live)
- Bonus imitation dalle sessioni umane
- Validazione del miglior genoma su una suite di seed fissi
- Persistenza (salva/carica genomi, record di validazione e checkpoint)
- Modalita' "watch" per vedere il miglior genoma giocare
"""

import datetime
import glob
import gzip
import json
import math
import os
import pickle
import random
import time
import traceback
import multiprocessing
import neat
import pygame
from settings import SCREEN_WIDTH, SCREEN_HEIGHT, FPS
from src.game_env import GameEnv

AI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ai")
CONFIG_PATH = os.path.join(AI_DIR, "neat_config.txt")
BEST_GENOME_PATH = os.path.join(AI_DIR, "best_genome.pkl")
# Record di validazione di best_genome.pkl (media, seed, versione della fitness)
BEST_GENOME_META_PATH = os.path.join(AI_DIR, "best_genome_meta.json")
CHECKPOINT_PREFIX = os.path.join(AI_DIR, "neat-checkpoint-")
MAX_CHECKPOINTS = 2  # mantieni solo gli ultimi N checkpoint
SESSIONS_DIR = os.path.join(AI_DIR, "human_sessions")

# Durate in secondi di gioco, convertite in frame con settings.FPS (GameEnv
# avanza di 1000/FPS ms a ogni step)
MAX_EPISODE_SECONDS = 180  # 3 minuti
MIN_EPISODE_SECONDS = 30   # alla generazione 0, poi cresce fino a MAX in 50 gen
MAX_FRAMES_PER_EPISODE = int(MAX_EPISODE_SECONDS * FPS)  # 10800 a 60 FPS
MIN_FRAMES_PER_EPISODE = int(MIN_EPISODE_SECONDS * FPS)  # 1800 a 60 FPS

# Velocita' di rendering durante il training visuale
TRAIN_RENDER_SKIP = 4  # mostra 1 frame su 4

# Popolazione del training headless (il visuale usa pop_size di neat_config.txt)
HEADLESS_POP_SIZE = 200

# Episodi per genoma: stessi seed per tutti i genomi della generazione
# (common random numbers), nuovi a ogni generazione; fitness = media
EPISODES_PER_GENOME = 2

# Terminazione anticipata genomi scarsi (check periodico, in secondi di gioco)
EARLY_STOP_INTERVAL = int(5 * FPS)   # controlla ogni 5 secondi
EARLY_STOP_MIN_FRAME = int(6 * FPS)  # non terminare prima di 6 secondi
EARLY_STOP_MAX_LIVES_LOST = 2  # 2 vite perse su PLAYER_LIVES (3): interrompi
EARLY_STOP_LIVES_GEN = 10      # nelle prime generazioni non terminare per vite
EARLY_STOP_IDLE_RATIO = 0.70   # se fermo per il 70% del tempo, interrompi

# Imitation bonus: generazione di attivazione, tetto, sessioni usate
IMITATION_START_GEN = 10
IMITATION_MAX_BONUS = 300
IMITATION_TOP_SESSIONS = 5           # le sessioni con lo score piu' alto
IMITATION_SAMPLES_PER_SESSION = 200  # frame equidistanti per sessione
IMITATION_WEIGHT_CAP = 4.0           # peso = min(tetto, ln(1 + score / 1000))

# Versione della fitness: incrementarla quando cambiano _compute_fitness, le
# regole di GameEnv o la suite di validazione. Il record di validazione
# salvato con un'altra versione non e' confrontabile e viene ricalcolato;
# riprendendo un checkpoint di un'altra versione si azzera lo storico di
# fitness delle specie (vedi NeatTrainer._reset_species_history)
FITNESS_VERSION = 2

# Validazione del best: a fine generazione i VALIDATION_TOP_K genomi con la
# fitness di gioco piu' alta giocano un episodio intero (MAX_FRAMES_PER_EPISODE,
# niente early stop, niente imitation) per ogni seed fisso; best_genome.pkl
# e' il genoma con la media piu' alta, se batte il record
VALIDATION_TOP_K = 5
VALIDATION_SEEDS = tuple(range(10 ** 9, 10 ** 9 + 10))  # fuori dalla serie del training


def _compute_fitness(env):
    """Calcola fitness e statistiche da un GameEnv dopo la simulazione.

    Funzione top-level (picklable) usata sia dal worker che da _eval_single.
    I termini a tempo sono in secondi di gioco (frame / FPS), quindi i pesi
    non dipendono da settings.FPS. La fitness e' la somma dei componenti
    comp_* di env_stats:

        (uccisioni a proiettile + da missile + da bomba) x 50
        + accuracy x uccisioni a proiettile x 10
        + schivate x 5 (passaggi ravvicinati senza collisione)
        + secondi x 1.5 - vite perse x 200 + livello^2 x 50
        + power-up x 50 + missili raccolti x 30 + livelli upgrade x 20
        - colpi allo scudo x 15 - scudi rotti x 120
        - secondi fermo x 15 + near-miss x 3
        + saldo netto (>= 0) dei secondi in avvicinamento a un power-up x 3
        - raffiche x 0.01

    Ritorna (fitness, env_stats).
    """
    accuracy_bonus = 0.0
    if env.shots_fired > 0:
        accuracy = env.shots_hit / env.shots_fired
        # Solo le uccisioni a proiettile: l'accuracy misura i proiettili
        accuracy_bonus = accuracy * env.enemies_killed * 10

    components = {
        'comp_kills': env.enemies_killed * 50,
        'comp_missile_kills': env.missile_kills * 50,
        'comp_bomb_kills': env.bomb_kills * 50,
        'comp_dodges': env.dodges * 5,
        'comp_accuracy': accuracy_bonus,
        'comp_frames': env.frame_count / FPS * 1.5,
        'comp_lives': -env.lives_lost * 200,
        'comp_level': env.level_manager.level ** 2 * 50,
        'comp_powerups': env.powerups_collected * 50,
        'comp_missiles': env.missiles_collected * 30,
        'comp_upgrades': env.total_upgrade_levels * 20,
        'comp_shield': -(env.shield_hits * 15 + env.shield_breaks * 120),
        'comp_idle': -(env.idle_frames / FPS * 15),
        'comp_near_misses': env.near_misses * 3,
        'comp_pu_approach': max(0, env.powerup_approach_frames) / FPS * 3,
        'comp_shoot_penalty': -env.shots_fired * 0.01,
    }
    # NON clippare a 0: i genomi pessimi devono avere fitness negativa
    fitness = sum(components.values())

    env_stats = {
        'score': env.score,
        'frame_count': env.frame_count,
        'level': env.level_manager.level,
        'lives_lost': env.lives_lost,
        'powerups': env.powerups_collected,
        'missiles': env.missiles_collected,
        'upgrades': env.total_upgrade_levels,
        'idle_frames': env.idle_frames,
        'enemies_killed': env.enemies_killed,
        'missile_kills': env.missile_kills,
        'bomb_kills': env.bomb_kills,
        'dodges': env.dodges,
        'accuracy': env.shots_hit / max(1, env.shots_fired),
        'near_misses': env.near_misses,
        **components,
    }

    return fitness, env_stats


def _empty_env_stats():
    """Statistiche di un genoma la cui valutazione e' fallita (tutte a zero)."""
    stats = {
        'score': 0, 'frame_count': 0, 'level': 1, 'lives_lost': 0,
        'powerups': 0, 'missiles': 0, 'upgrades': 0, 'idle_frames': 0,
        'enemies_killed': 0, 'missile_kills': 0, 'bomb_kills': 0,
        'dodges': 0, 'accuracy': 0, 'near_misses': 0,
    }
    for key in ('comp_kills', 'comp_missile_kills', 'comp_bomb_kills',
                'comp_dodges', 'comp_accuracy', 'comp_frames', 'comp_lives',
                'comp_level', 'comp_powerups', 'comp_missiles', 'comp_upgrades',
                'comp_shield', 'comp_idle', 'comp_near_misses',
                'comp_pu_approach', 'comp_shoot_penalty'):
        stats[key] = 0
    return stats


def _mean_stats(stats_list):
    """Media chiave per chiave delle env_stats di piu' episodi."""
    n = len(stats_list)
    return {k: sum(s[k] for s in stats_list) / n for k in stats_list[0]}


def _generation_seeds(generation):
    """Seed comuni a tutti i genomi della generazione (common random numbers)."""
    return tuple(generation * 1000 + k for k in range(EPISODES_PER_GENOME))


def _early_stop(env, frame, generation):
    """Terminazione anticipata dei genomi scarsi (check periodico).

    Unica regola per il worker e per _eval_single (stessa fitness in headless
    e nel visuale). `frame` e' l'indice del frame appena giocato.
    """
    if frame < EARLY_STOP_MIN_FRAME or frame % EARLY_STOP_INTERVAL != 0:
        return False
    # Nelle prime generazioni non terminare per vite perse (servono dati)
    if generation >= EARLY_STOP_LIVES_GEN and env.lives_lost >= EARLY_STOP_MAX_LIVES_LOST:
        return True
    # Troppo inattivo: fermo (anche premendo contro il bordo) quasi sempre
    return env.idle_frames / max(1, frame) >= EARLY_STOP_IDLE_RATIO


def _decode_actions_static(output):
    """Converte output della rete in azioni (usata dai worker)."""
    v = output[0]
    if v < -0.2:
        vertical = -1
    elif v > 0.2:
        vertical = 1
    else:
        vertical = 0
    h = output[1]
    if h < -0.2:
        horizontal = -1
    elif h > 0.2:
        horizontal = 1
    else:
        horizontal = 0
    return {
        'vertical': vertical,
        'horizontal': horizontal,
        'shoot': output[2] > 0.3,
        'missile': output[3] > 0.5,
        'bomb': output[4] > 0.5,
    }


def compile_network(genome, config):
    """Compila la rete feed-forward del genoma in una funzione Python generata.

    Equivale bit per bit a neat.nn.FeedForwardNetwork.activate ma e' circa 2x
    piu' veloce: stessi nodi nello stesso ordine, stessi termini peso*input
    passati alla stessa funzione di aggregazione (su una tupla invece che su
    una lista), stessa espressione bias + response * s. Non sommare i termini
    inline (a + b + c): da Python 3.12 sum() sui float e' compensata e il
    risultato cambierebbe negli ultimi bit.

    Restituisce act(inputs) -> lista degli output. La funzione non e'
    picklable: non salvarla sul genoma (finirebbe nei checkpoint e in
    best_genome.pkl).
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    in_idx = {k: j for j, k in enumerate(net.input_nodes)}
    # repr() di inf/nan produce un nome, non un letterale
    ns = {'inf': math.inf, 'nan': math.nan}

    def ref(k):
        if k in in_idx:
            return f"x[{in_idx[k]}]"
        return f"n_{'m' if k < 0 else 'p'}{abs(k)}"

    # feed_forward_layers garantisce che ogni link referenzi un input o un
    # nodo gia' calcolato nelle righe precedenti
    lines = ["def act(x):"]
    computed = set()
    for j, (node, act_func, agg_func, bias, response, links) in enumerate(net.node_evals):
        ns[f"A{j}"] = act_func
        ns[f"G{j}"] = agg_func
        terms = "".join(f"{ref(i)} * {w!r}, " for i, w in links)
        lines.append(f"    {ref(node)} = A{j}({bias!r} + {response!r} * G{j}(({terms})))")
        computed.add(node)
    # Output mai calcolati: restano a 0.0 come in FeedForwardNetwork.values
    outs = ", ".join(ref(k) if k in computed else "0.0" for k in net.output_nodes)
    lines.append(f"    return [{outs}]")
    exec(compile("\n".join(lines), "<neat-net>", "exec"), ns)
    return ns["act"]


# Componenti confrontate dall'imitation bonus e loro valori: missile e bomba
# sono esclusi (nelle sessioni umane sono pressioni rarissime, quindi il
# confronto premierebbe solo il non usarli mai)
_IMITATION_COMPONENTS = {
    'vertical': (-1, 0, 1),
    'horizontal': (-1, 0, 1),
    'shoot': (False, True),
}


def _component_agreement(component, ai_value, human_value):
    """Accordo (0-1) su una componente: assi 1 / 0.5 (adiacente) / 0, sparo 1 / 0."""
    if component == 'shoot':
        return 1.0 if ai_value == human_value else 0.0
    diff = abs(ai_value - human_value)
    if diff == 0:
        return 1.0
    return 0.5 if diff == 1 else 0.0


def _action_frequencies(samples, actions=None):
    """Frequenze pesate dei valori di ogni componente.

    Senza `actions` sono quelle delle azioni umane dei campioni, altrimenti
    quelle di `actions` (una per campione, stesso ordine). None se il peso
    totale e' zero.
    """
    total = sum(w for _, _, w in samples)
    if total <= 0:
        return None
    freq = {c: dict.fromkeys(values, 0.0) for c, values in _IMITATION_COMPONENTS.items()}
    for i, (_obs, human_action, weight) in enumerate(samples):
        action = human_action if actions is None else actions[i]
        for c in freq:
            freq[c][action[c]] += weight / total
    return freq


def _chance_similarity(ai_freq, human_freq):
    """Somiglianza attesa tra due policy indipendenti dall'osservazione.

    E' la somiglianza che otterrebbe una policy con le stesse frequenze
    d'azione del genoma ma scelte a caso rispetto allo stato: per una policy
    costante coincide con la sua somiglianza effettiva.
    """
    total = 0.0
    for c, values in _IMITATION_COMPONENTS.items():
        total += sum(
            _component_agreement(c, a, h) * ai_freq[c][a] * human_freq[c][h]
            for a in values for h in values
        )
    return total / len(_IMITATION_COMPONENTS)


def _imitation_bonus_from_samples(act, samples, human_freq):
    """Bonus fitness per somiglianza alle azioni umane registrate (0-300).

    `act` e' la rete compilata del genoma, `samples` la lista di tuple
    (obs, azione umana, peso) e `human_freq` le frequenze pesate delle azioni
    umane, entrambi da NeatTrainer._load_imitation_samples. Conta solo la
    somiglianza oltre quella dovuta alle frequenze d'azione del genoma
    (kappa di Cohen pesato):
        bonus = max(0, sim - sim_caso) / (1 - sim_caso) * 300
    Ogni policy costante (ferma, sempre a sparare...) e ogni policy casuale
    indipendente dallo stato prendono 0: rende solo agire come l'umano
    nelle stesse situazioni.
    """
    if not samples or human_freq is None:
        return 0.0
    actions = [_decode_actions_static(act(obs)) for obs, _, _ in samples]
    total_match = 0.0
    total_weight = 0.0
    for action, (_obs, human_action, weight) in zip(actions, samples):
        total_match += NeatTrainer._action_similarity(action, human_action) * weight
        total_weight += weight
    similarity = total_match / total_weight
    chance = _chance_similarity(_action_frequencies(samples, actions), human_freq)
    # Margine minimo: per una policy costante sim e sim_caso coincidono a
    # meno dell'arrotondamento
    if chance >= 1.0 or similarity - chance <= 1e-9:
        return 0.0
    return (similarity - chance) / (1.0 - chance) * IMITATION_MAX_BONUS


# Imitation bonus nel processo worker (impostati da _init_worker)
_IMITATION_SAMPLES = None
_IMITATION_HUMAN_FREQ = None


def _init_worker(imitation_samples=None, imitation_human_freq=None):
    """Inizializza pygame in ogni processo worker del Pool.

    `imitation_samples` e `imitation_human_freq` sono estratti in train()
    prima di creare il Pool: il worker calcola l'imitation bonus in parallelo
    invece di lasciarlo in serie al processo principale.
    """
    global _IMITATION_SAMPLES, _IMITATION_HUMAN_FREQ
    _IMITATION_SAMPLES = imitation_samples
    _IMITATION_HUMAN_FREQ = imitation_human_freq
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"  # i worker non aprono il device audio
    pygame.init()
    pygame.display.set_mode((1, 1))
    # Imposta priorità bassa per non saturare le CPU
    try:
        import ctypes
        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def _run_episode(act, seed, max_frames, generation=None):
    """Gioca un episodio headless e ritorna (fitness, env_stats).

    Con `generation` applica l'early stop del training (_early_stop); con
    None (suite di validazione) l'episodio dura fino al game over o a
    max_frames, come una partita in coop.
    """
    env = GameEnv(render=False, screen=None)
    obs = env.reset(seed=seed)
    for frame in range(max_frames):
        obs, reward, done = env.step(_decode_actions_static(act(obs)))
        if done:
            break
        if generation is not None and _early_stop(env, frame, generation):
            break
    result = _compute_fitness(env)
    env.close()
    return result


def _eval_genome_worker(args):
    """Valuta un singolo genoma in un processo separato (per multiprocessing).

    Funzione top-level per essere picklable. Gioca un episodio per ogni seed
    della generazione. Restituisce (idx, fitness, env_stats, imitation_bonus)
    per imap_unordered: fitness e statistiche sono la media sugli episodi.
    """
    idx, genome, config, max_frames, generation, seeds = args

    try:
        act = compile_network(genome, config)
        results = [_run_episode(act, seed, max_frames, generation) for seed in seeds]
        fitness = sum(f for f, _ in results) / len(results)
        env_stats = _mean_stats([s for _, s in results])

        # Imitation bonus calcolato qui, in parallelo, invece che nel main
        imitation_bonus = 0.0
        if _IMITATION_SAMPLES is not None and generation >= IMITATION_START_GEN:
            imitation_bonus = _imitation_bonus_from_samples(
                act, _IMITATION_SAMPLES, _IMITATION_HUMAN_FREQ
            )
        return idx, fitness, env_stats, imitation_bonus
    except Exception:
        # Fitness molto negativa per non bloccare il training, ma con traccia
        # in console: un errore sistematico deve essere visibile
        print(f"[worker] valutazione del genoma {getattr(genome, 'key', '?')} "
              f"fallita:\n{traceback.format_exc()}", flush=True)
        return idx, -10000.0, _empty_env_stats(), 0.0


def _validation_worker(args):
    """Un episodio della suite di validazione (top-level per il Pool).

    Restituisce (key, seed, fitness, env_stats); fitness None se la
    valutazione fallisce (genoma incompatibile).
    """
    key, genome, config, seed, max_frames = args
    try:
        act = compile_network(genome, config)
        fitness, env_stats = _run_episode(act, seed, max_frames)
        return key, seed, fitness, env_stats
    except Exception:
        print(f"[validazione] genoma {key}, seed {seed} fallito:\n"
              f"{traceback.format_exc()}", flush=True)
        return key, seed, None, None


def _list_checkpoints(prefix=None):
    """Checkpoint su disco come lista ordinata di (generazione, path).

    Ignora i file con suffisso non numerico (es. neat-checkpoint-80.bak o il
    temporaneo di scrittura neat-checkpoint-tmp).
    """
    prefix = prefix or CHECKPOINT_PREFIX
    base = os.path.basename(prefix)
    found = []
    for path in glob.glob(glob.escape(prefix) + "*"):
        suffix = os.path.basename(path)[len(base):]
        if suffix.isdigit():
            found.append((int(suffix), path))
    found.sort()
    return found


class _RollingCheckpointer(neat.Checkpointer):
    """Checkpointer atomico che mantiene solo gli ultimi MAX_CHECKPOINTS file."""

    def save_checkpoint(self, config, population, species_set, generation):
        """Stesso contenuto e formato di neat.Checkpointer.save_checkpoint
        (neat-python 1.1.0, letto da restore_checkpoint), ma scritto su un file
        temporaneo e poi rinominato: un'interruzione durante la scrittura non
        lascia un checkpoint troncato con il numero piu' alto.
        """
        filename = f'{self.filename_prefix}{generation}'
        tmp = f'{self.filename_prefix}tmp'
        print(f"Saving checkpoint to {filename}")
        with gzip.open(tmp, 'w', compresslevel=5) as f:
            data = (generation, config, population, species_set, random.getstate())
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, filename)
        self._cleanup_old_checkpoints(self.filename_prefix, generation)

    @staticmethod
    def _cleanup_old_checkpoints(prefix=None, saved_generation=None):
        """Tiene gli ultimi MAX_CHECKPOINTS fino a quello appena salvato.

        I file con numero piu' alto (illeggibili, saltati alla ripresa) non
        contano: altrimenti la rotazione cancellerebbe checkpoint validi.
        """
        files = _list_checkpoints(prefix)
        if saved_generation is not None:
            files = [(g, p) for g, p in files if g <= saved_generation]
        if len(files) <= MAX_CHECKPOINTS:
            return
        for _, old in files[:-MAX_CHECKPOINTS]:
            try:
                os.remove(old)
            except OSError:
                pass


class NeatTrainer:
    """Gestisce il ciclo di vita NEAT: training, salvataggio, replay."""

    def __init__(self, screen=None):
        self.screen = screen
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            CONFIG_PATH,
        )
        self.best_genome = None
        # Record di validazione: media della suite (VALIDATION_SEEDS) del
        # genoma in best_genome.pkl, -inf se non ancora stabilito
        self.best_fitness = -math.inf
        self.generation = 0
        self._start_generation = 0  # generazione di partenza della sessione
        # Imitation bonus: campioni (obs, azione umana, peso) dalle sessioni
        # umane con lo score piu' alto e frequenze delle azioni umane,
        # caricati una volta in train() prima del Pool
        self._imitation_samples = None
        self._imitation_human_freq = None
        self._human_sessions_count = None  # None = sessioni non ancora caricate
        self._headless = False
        self._font = None
        self._font_small = None
        self._font_big = None
        self._font_section = None
        # Riferimenti NEAT (set in train())
        self._population = None
        self._stats_reporter = None
        self._checkpointer = None
        # Statistiche per schermata headless
        self._last_gen_stats = None
        self._current_gen_fitnesses = []
        self._training_start_time = None
        self._gen_total = 0             # genomi della generazione corrente
        self._genomes_evaluated = 0     # genomi valutati in questa sessione
        # Storico generazioni
        self._fitness_history = []       # list of (avg, best, stdev)
        self._species_count_history = []
        self._gen_times = []
        self._gen_start_time = None
        # Metriche GameEnv aggregate per gen corrente
        self._current_gen_scores = []
        self._current_gen_levels = []
        self._current_gen_frames = []
        self._current_gen_lives_lost = []
        self._current_gen_powerups = []
        self._current_gen_missiles = []
        self._current_gen_upgrades = []
        self._current_gen_idle_frames = []
        # Scomposizione fitness: genoma record (medie sulla suite di
        # validazione) e genomi della gen corrente (stesso ordine di
        # _current_gen_fitnesses)
        self._best_genome_components = None
        self._current_gen_fitness_components = []
        self._new_record_this_gen = False
        # Validazione: cache per genome.key (media, componenti medi, fitness
        # per seed), valida per tutta la sessione (suite e fitness non
        # cambiano), e progresso
        self._val_cache = {}
        self._last_validation = None    # (key, media) del migliore della gen
        self._validation_progress = None
        self._pool = None

    def _init_fonts(self):
        if self._font is None:
            self._font = pygame.font.Font(None, 32)
            self._font_small = pygame.font.Font(None, 24)
            self._font_big = pygame.font.Font(None, 56)
            self._font_section = pygame.font.Font(None, 28)

    def train(self, generations=99999, headless=False):
        """Avvia o riprende il training NEAT.

        Args:
            headless: se True, niente rendering del gioco, solo stats a schermo.
        """
        self._headless = headless
        self._training_start_time = time.time()
        self._init_fonts()

        # In headless: popolazione piu' grande per piu' diversita'. Va
        # impostata prima del restore, che riparte da self.config
        if headless:
            self.config.pop_size = HEADLESS_POP_SIZE
        # Versione della fitness, salvata nei checkpoint con la config
        self.config.fitness_version = FITNESS_VERSION

        # Cerca checkpoint per riprendere
        checkpoint, saved_config = self._find_latest_checkpoint()
        if checkpoint:
            print(f"Ripresa dal checkpoint: {checkpoint}")
            # Il compatibility threshold e' stato del controllo di speciazione
            # (_check_and_fix_stagnation), non un iperparametro: si riprende il
            # valore adattato nel checkpoint, quello del file vale per un run nuovo
            threshold = saved_config.species_set_config.compatibility_threshold
            self.config.species_set_config.compatibility_threshold = threshold
            # new_config: pop_size, mutazioni, stagnazione e riproduzione da
            # neat_config.txt (senza, neat usa la config pickled nel checkpoint)
            p = neat.Checkpointer.restore_checkpoint(checkpoint, new_config=self.config)
            # Lo species set ripristinato ha ancora la species_set_config del
            # pickle: senza questo il threshold non agirebbe sulla speciazione
            p.species.species_set_config = p.config.species_set_config
            # Sincronizza generazione interna con il checkpoint
            self.generation = p.generation
            print(f"  Generazione ripristinata: {self.generation}, "
                  f"pop_size {p.config.pop_size}, threshold {threshold:.3f}")
            saved_version = getattr(saved_config, 'fitness_version', 1)
            if saved_version != FITNESS_VERSION:
                self._reset_species_history(p, saved_version)
        else:
            print(f"Inizio nuovo training (pop_size={self.config.pop_size})")
            p = neat.Population(self.config)
        self._start_generation = self.generation

        # Reporter
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        self._checkpointer = _RollingCheckpointer(
            generation_interval=10,
            filename_prefix=CHECKPOINT_PREFIX,
        )
        p.add_reporter(self._checkpointer)

        self._population = p
        self._stats_reporter = stats

        # Sessioni umane per l'imitation bonus: caricate prima del Pool,
        # perche' i campioni vengono passati ai worker con initargs
        self._load_imitation_samples()

        # Pool persistente per headless (evita di ricreare i processi ogni gen)
        if headless:
            num_workers = max(1, multiprocessing.cpu_count() - 1)
            self._pool = multiprocessing.Pool(
                processes=num_workers, initializer=_init_worker,
                initargs=(self._imitation_samples, self._imitation_human_freq),
            )

        try:
            # Record di validazione di best_genome.pkl (sidecar o rivalutazione)
            self._init_validation_record(p.config)
            p.run(self._eval_genomes, generations)
        except _TrainingInterrupted:
            print("\nTraining interrotto dall'utente.")
            self._save_interrupt_checkpoint(p)
        except KeyboardInterrupt:
            self._save_interrupt_checkpoint(p)
            raise
        finally:
            if self._pool:
                self._pool.terminate()
                self._pool.join()
                self._pool = None

        # Il miglior genoma e' gia' salvato incrementalmente in _eval_genomes
        if self.best_genome and self.best_fitness > -math.inf:
            print(f"\nRecord di validazione: {self.best_fitness:.1f}")

    @staticmethod
    def _reset_species_history(p, saved_version):
        """Azzera lo storico di fitness delle specie dopo un cambio di fitness.

        DefaultStagnation considera migliorata una specie solo se supera il
        massimo del suo storico: con lo storico nella vecchia scala nessuna
        specie migliorerebbe e dopo max_stagnation generazioni verrebbero
        eliminate tutte tranne le species_elitism protette.
        """
        for s in p.species.species.values():
            s.fitness_history = []
            s.last_improved = p.generation
            s.fitness = None
            s.adjusted_fitness = None
        print(f"  Fitness del checkpoint v{saved_version}, attuale v{FITNESS_VERSION}: "
              f"storico di {len(p.species.species)} specie azzerato")

    def _save_interrupt_checkpoint(self, p):
        """Checkpoint all'interruzione (ESC, chiusura, Ctrl+C).

        Salva la popolazione della generazione in corso, prima della sua
        valutazione (lo stesso stato del checkpoint periodico con quel
        numero). Solo se in questa sessione e' stata completata almeno una
        generazione: altrimenti lo stato coincide con il checkpoint ripreso.
        """
        if self._checkpointer is None or p.generation <= self._start_generation:
            return
        try:
            self._checkpointer.save_checkpoint(p.config, p.population, p.species, p.generation)
        except Exception as e:
            print(f"Checkpoint di interruzione non salvato: {e!r}")

    def _eval_genomes(self, genomes, config):
        """Valuta tutti i genomi della popolazione.

        In modalita' headless usa multiprocessing per parallelizzare. Ogni
        genoma gioca un episodio per seed della generazione (gli stessi per
        tutti); a fine generazione i migliori passano dalla validazione.
        """
        # Sessioni umane (di norma gia' caricate in train())
        if self._human_sessions_count is None:
            self._load_imitation_samples()

        self._gen_start_time = time.time()

        total = len(genomes)
        self._gen_total = total
        render = self.screen is not None and not self._headless

        # Reset liste per gen corrente
        self._current_gen_fitnesses = []
        self._current_gen_scores = []
        self._current_gen_levels = []
        self._current_gen_frames = []
        self._current_gen_lives_lost = []
        self._current_gen_powerups = []
        self._current_gen_missiles = []
        self._current_gen_upgrades = []
        self._current_gen_idle_frames = []
        self._current_gen_fitness_components = []
        game_fitness = {}  # idx -> fitness di gioco (senza imitation)

        max_frames = self._get_max_frames()
        seeds = _generation_seeds(self.generation)

        if self._headless:
            # ── Valutazione PARALLELA (headless, pool persistente) ──
            worker_args = [
                (idx, genome, config, max_frames, self.generation, seeds)
                for idx, (genome_id, genome) in enumerate(genomes)
            ]

            # imap_unordered: i worker non restano mai idle in attesa
            # dei genomi piu' lenti — appena uno finisce, parte il successivo
            completed = 0
            for idx, fitness, env_stats, imitation_bonus in self._pool.imap_unordered(
                _eval_genome_worker, worker_args, chunksize=1
            ):
                genome_id, genome = genomes[idx]
                genome.fitness = fitness
                game_fitness[idx] = fitness
                self._collect_stats(genome, env_stats, imitation_bonus)

                completed += 1
                # Aggiorna dashboard durante la valutazione
                if self.screen and completed % 5 == 0:
                    self._draw_stats_screen(completed, total, config)
                    pygame.display.flip()
                    self._poll_interrupt()

            # Aggiorna schermata finale
            if self.screen:
                self._draw_stats_screen(total, total, config)
                pygame.display.flip()
                self._poll_interrupt()
        else:
            # ── Valutazione SERIALE (con rendering) ──
            for idx, (genome_id, genome) in enumerate(genomes):
                fitness, env_stats = self._eval_single(
                    genome, config, idx + 1, total, render, seeds, max_frames
                )
                genome.fitness = fitness
                game_fitness[idx] = fitness
                self._collect_stats(genome, env_stats)

        best_fitness_in_gen = max(self._current_gen_fitnesses) if self._current_gen_fitnesses else 0.0

        # Validazione dei migliori: salva best_genome.pkl se batte il record
        self._new_record_this_gen = self._validate_top_genomes(genomes, game_fitness, config)

        # Statistiche fine generazione
        n = max(1, len(self._current_gen_fitnesses))
        avg_fit = sum(self._current_gen_fitnesses) / n
        stdev_fit = 0.0
        if n > 1:
            variance = sum((f - avg_fit) ** 2 for f in self._current_gen_fitnesses) / (n - 1)
            stdev_fit = variance ** 0.5

        gen_time = time.time() - self._gen_start_time
        self._gen_times.append(gen_time)
        self._fitness_history.append((avg_fit, best_fitness_in_gen, stdev_fit))

        # Conteggio specie
        num_species = 0
        if self._population and hasattr(self._population, 'species'):
            num_species = len(self._population.species.species)
        self._species_count_history.append(num_species)

        self._last_gen_stats = {
            'avg_fitness': avg_fit,
            'best_fitness': best_fitness_in_gen,
            'stdev': stdev_fit,
            'avg_score': sum(self._current_gen_scores) / n,
            'avg_level': sum(self._current_gen_levels) / n,
            'avg_frames': sum(self._current_gen_frames) / n,
            'avg_lives_lost': sum(self._current_gen_lives_lost) / n,
            'gen_time': gen_time,
        }

        self.generation += 1

        # Controllo speciazione (threshold adattivo)
        self._check_and_fix_stagnation(config)

        # In headless: aggiorna schermata fine generazione
        if self._headless and self.screen:
            self._draw_stats_screen(total, total, config)
            pygame.display.flip()

    def _poll_interrupt(self):
        """Svuota la coda eventi: ESC o chiusura finestra interrompono il training."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise _TrainingInterrupted()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                raise _TrainingInterrupted()

    # ── Validazione e persistenza del best ──

    def _validate_top_genomes(self, genomes, game_fitness, config):
        """Rivaluta i VALIDATION_TOP_K genomi con la fitness di gioco piu' alta
        sulla suite di validazione e salva il migliore se batte il record.

        I genomi gia' validati in questa sessione (elite riportate identiche
        alla generazione dopo) escono dalla cache. Ritorna True se nuovo record.
        """
        if not game_fitness:
            return False
        ranked = sorted(game_fitness, key=lambda i: game_fitness[i], reverse=True)
        candidates = [genomes[i][1] for i in ranked[:VALIDATION_TOP_K]]
        todo = [g for g in candidates if g.key not in self._val_cache]
        if todo:
            self._val_cache.update(self._run_validation(todo, config))
        best = max(candidates, key=lambda g: self._val_cache[g.key][0])
        mean, comps, per_seed = self._val_cache[best.key]
        self._last_validation = (best.key, mean)
        print(f"  [Validazione] top-{len(candidates)} ({len(todo)} nuovi): migliore genoma "
              f"{best.key}, media {mean:.1f} su {len(VALIDATION_SEEDS)} seed "
              f"(record {self.best_fitness:.1f})")
        if mean <= self.best_fitness:
            return False
        self.best_fitness = mean
        self.best_genome = best
        self._best_genome_components = comps
        self._save_best_genome(best, mean, per_seed)
        return True

    def _run_validation(self, genomes, config):
        """Gioca la suite di validazione per ogni genoma.

        In headless usa il Pool (un task per genoma e seed), nel visuale gira
        nel processo principale senza rendering. Ritorna {key: (media,
        componenti medi, [fitness per seed in ordine di seed])}; media -inf se
        un episodio fallisce (genoma incompatibile).
        """
        max_frames = MAX_FRAMES_PER_EPISODE
        tasks = [(g.key, g, config, seed, max_frames) for g in genomes for seed in VALIDATION_SEEDS]
        per_key = {g.key: {} for g in genomes}
        if self._pool is not None:
            results = self._pool.imap_unordered(_validation_worker, tasks, chunksize=1)
        else:
            results = (_validation_worker(t) for t in tasks)
        done = 0
        try:
            for key, seed, fitness, env_stats in results:
                per_key[key][seed] = (fitness, env_stats)
                done += 1
                self._validation_progress = (done, len(tasks))
                if self.screen is not None:
                    if self._headless:
                        self._draw_stats_screen(self._gen_total, self._gen_total, config)
                    else:
                        self._draw_validation_overlay(done, len(tasks))
                    pygame.display.flip()
                    self._poll_interrupt()
        finally:
            self._validation_progress = None

        out = {}
        for key, by_seed in per_key.items():
            # In ordine di seed: la media non dipende dall'ordine di arrivo
            ordered = [by_seed[s] for s in VALIDATION_SEEDS]
            fits = [f for f, _ in ordered]
            if any(f is None for f in fits):
                out[key] = (-math.inf, None, fits)
            else:
                out[key] = (sum(fits) / len(fits), _mean_stats([s for _, s in ordered]), fits)
        return out

    def _init_validation_record(self, config):
        """Fissa il record di validazione all'avvio.

        Legge BEST_GENOME_META_PATH. Se manca, e' illeggibile, si riferisce a
        un altro genoma o a un'altra versione della fitness/suite, rivaluta
        best_genome.pkl con la suite corrente e riscrive il sidecar (il pkl non
        cambia). Senza best_genome.pkl caricabile il record resta -inf e il
        primo genoma validato viene salvato.
        """
        self.best_fitness = -math.inf
        if not os.path.exists(BEST_GENOME_PATH):
            return
        try:
            with open(BEST_GENOME_PATH, 'rb') as f:
                self.best_genome = pickle.load(f)
        except Exception as e:
            print(f"best_genome.pkl non caricabile, record azzerato: {e!r}")
            self.best_genome = None
            return

        meta = self._read_best_meta()
        if meta is not None:
            self.best_fitness = float(meta['validation_mean'])
            print(f"Record di validazione: {self.best_fitness:.1f} ({BEST_GENOME_META_PATH})")
            return

        print("Rivalutazione di best_genome.pkl sulla suite di validazione "
              f"({len(VALIDATION_SEEDS)} seed, fitness v{FITNESS_VERSION})...")
        mean, comps, per_seed = self._run_validation([self.best_genome], config)[self.best_genome.key]
        if mean == -math.inf:
            print("  best_genome.pkl non valutabile: record azzerato")
            return
        self.best_fitness = mean
        self._best_genome_components = comps
        print(f"  Record di validazione: {mean:.1f}")
        self._write_best_meta(self.best_genome, mean, per_seed)

    def _read_best_meta(self):
        """Sidecar del best se valido per il genoma e la suite correnti, altrimenti None."""
        try:
            with open(BEST_GENOME_META_PATH, encoding='utf-8') as f:
                meta = json.load(f)
            float(meta['validation_mean'])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if (meta.get('fitness_version') != FITNESS_VERSION
                or meta.get('validation_seeds') != list(VALIDATION_SEEDS)
                or meta.get('max_frames') != MAX_FRAMES_PER_EPISODE
                or meta.get('genome_key') != getattr(self.best_genome, 'key', None)):
            return None
        return meta

    def _write_best_meta(self, genome, mean, per_seed):
        """Scrive il sidecar del best (JSON, atomico: .tmp + os.replace)."""
        meta = {
            'fitness_version': FITNESS_VERSION,
            'validation_mean': mean,
            'validation_seeds': list(VALIDATION_SEEDS),
            'validation_fitness': per_seed,
            'max_frames': MAX_FRAMES_PER_EPISODE,
            'genome_key': genome.key,
            'generation': self.generation,
            'saved_at': datetime.datetime.now().isoformat(timespec='seconds'),
        }
        os.makedirs(os.path.dirname(BEST_GENOME_META_PATH), exist_ok=True)
        tmp = BEST_GENOME_META_PATH + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=1)
        os.replace(tmp, BEST_GENOME_META_PATH)

    def _save_best_genome(self, genome, mean, per_seed):
        """Salva best_genome.pkl (stesso formato: il genoma pickled) e il sidecar."""
        os.makedirs(AI_DIR, exist_ok=True)
        tmp = BEST_GENOME_PATH + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(genome, f)
        os.replace(tmp, BEST_GENOME_PATH)
        self._write_best_meta(genome, mean, per_seed)

    def _get_max_frames(self):
        """Frame massimi adattivi: piu' generazioni = piu' tempo concesso."""
        # Cresce da MIN a MAX (30 s -> 3 min di gioco) in 50 generazioni
        t = min(1.0, self.generation / 50.0)
        return int(MIN_FRAMES_PER_EPISODE + t * (MAX_FRAMES_PER_EPISODE - MIN_FRAMES_PER_EPISODE))

    def _collect_stats(self, genome, env_stats, imitation_bonus=None):
        """Raccoglie le statistiche di un genoma valutato.

        Applica anche l'imitation bonus se disponibile: in headless arriva
        gia' calcolato dal worker, nel training visuale viene calcolato qui.
        Il bonus entra nei componenti come 'comp_imitation'.
        """
        bonus = 0.0
        if self._imitation_samples is not None and self.generation >= IMITATION_START_GEN:
            if imitation_bonus is None:
                imitation_bonus = self._imitation_bonus(genome, self.config)
            bonus = imitation_bonus
            genome.fitness += bonus
        env_stats = dict(env_stats, comp_imitation=bonus)
        self._genomes_evaluated += 1
        self._current_gen_fitnesses.append(genome.fitness)
        self._current_gen_scores.append(env_stats['score'])
        self._current_gen_levels.append(env_stats['level'])
        self._current_gen_frames.append(env_stats['frame_count'])
        self._current_gen_lives_lost.append(env_stats['lives_lost'])
        self._current_gen_powerups.append(env_stats['powerups'])
        self._current_gen_missiles.append(env_stats['missiles'])
        self._current_gen_upgrades.append(env_stats['upgrades'])
        self._current_gen_idle_frames.append(env_stats['idle_frames'])
        self._current_gen_fitness_components.append(env_stats)

    def _eval_single(self, genome, config, genome_num=0, total=0, render=False,
                     seeds=None, max_frames=None):
        """Valuta un genoma giocando un episodio per seed (media come il worker).

        Se render=True, mostra le partite sullo schermo in tempo reale.
        Stesse regole del worker (_run_episode): early stop con _early_stop.
        """
        act = compile_network(genome, config)
        if seeds is None:
            seeds = _generation_seeds(self.generation)
        if max_frames is None:
            max_frames = self._get_max_frames()
        results = []
        for ep, seed in enumerate(seeds):
            env = GameEnv(render=render, screen=self.screen)
            obs = env.reset(seed=seed)

            for frame in range(max_frames):
                # Gestione eventi (ESC per interrompere)
                if render:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            env.close()
                            raise _TrainingInterrupted()
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                env.close()
                                raise _TrainingInterrupted()

                output = act(obs)
                actions = self._decode_actions(output)
                # Disegna solo i frame che verranno mostrati (1 su TRAIN_RENDER_SKIP):
                # lo sfondo avanza comunque in _update, quindi l'immagine non cambia
                obs, reward, done = env.step(
                    actions, draw=render and frame % TRAIN_RENDER_SKIP == 0
                )

                # Rendering: mostra 1 frame su TRAIN_RENDER_SKIP
                if render and frame % TRAIN_RENDER_SKIP == 0:
                    self._draw_training_overlay(
                        env, genome_num, total, frame, ep + 1, len(seeds)
                    )
                    pygame.display.flip()

                # Game over o terminazione anticipata (genoma scarso)
                if done or _early_stop(env, frame, self.generation):
                    break

            results.append(_compute_fitness(env))
            env.close()

        fitness = sum(f for f, _ in results) / len(results)
        return fitness, _mean_stats([s for _, s in results])

    def _check_and_fix_stagnation(self, config):
        """Controllo della speciazione: adatta il compatibility threshold.

        Agisce sulla speciazione della generazione successiva perche' lo
        species set usa config.species_set_config (anche dopo il restore, vedi
        train()). La stagnazione delle specie la gestisce neat
        (DefaultStagnation: max_stagnation, species_elitism).
        """
        if not self._population:
            return

        species_set = self._population.species
        num_species = len(species_set.species)

        # Se troppo poche specie: abbassa threshold per forzare speciazione
        if num_species <= 2:
            threshold = config.species_set_config.compatibility_threshold
            if threshold > 1.0:
                new_t = max(1.0, threshold * 0.85)
                config.species_set_config.compatibility_threshold = new_t
                print(f"  [Anti-stagnazione] Threshold abbassato: {threshold:.2f} -> {new_t:.2f}")
        # Se troppe specie: alza threshold per evitare frammentazione
        elif num_species > 20:
            threshold = config.species_set_config.compatibility_threshold
            new_t = min(6.0, threshold * 1.1)
            config.species_set_config.compatibility_threshold = new_t
            print(f"  [Anti-stagnazione] Troppe specie ({num_species}), threshold alzato: {threshold:.2f} -> {new_t:.2f}")

    def _draw_training_overlay(self, env, genome_num, total, frame, episode=1, episodes=1):
        """Disegna l'overlay con info training sopra il gioco."""
        overlay_bg = pygame.Surface((SCREEN_WIDTH, 50), pygame.SRCALPHA)
        overlay_bg.fill((0, 0, 0, 160))
        self.screen.blit(overlay_bg, (0, 0))

        # Riga superiore: info generazione e genoma
        n_sessions = self._human_sessions_count or 0
        imitation = f" | Sessioni umane: {n_sessions}" if n_sessions > 0 else ""

        top_text = self._font.render(
            f"Gen {self.generation + 1} | "
            f"Genoma {genome_num}/{total} (ep. {episode}/{episodes}) | "
            f"Record val.: {self._format_record()}"
            f"{imitation}",
            True, (255, 255, 0)
        )
        self.screen.blit(top_text, (10, 4))

        # Score e stato del genoma corrente (in basso)
        overlay_bg2 = pygame.Surface((SCREEN_WIDTH, 30), pygame.SRCALPHA)
        overlay_bg2.fill((0, 0, 0, 140))
        self.screen.blit(overlay_bg2, (0, SCREEN_HEIGHT - 30))

        bottom_text = self._font_small.render(
            f"Score: {env.score} | "
            f"Vite: {env.player.lives} | "
            f"Livello: {env.level_manager.level} | "
            f"Frame: {frame} | "
            f"ESC=interrompi",
            True, (180, 220, 180)
        )
        self.screen.blit(bottom_text, (10, SCREEN_HEIGHT - 26))

    def _draw_validation_overlay(self, done, total):
        """Training visuale: avanzamento della validazione (senza rendering)."""
        self.screen.fill((10, 10, 30))
        text = self._font.render(
            f"Gen {self.generation + 1} | Validazione dei migliori: episodio {done}/{total} | "
            f"Record val.: {self._format_record()}",
            True, (255, 255, 0)
        )
        self.screen.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)))

    def _format_record(self):
        """Record di validazione per HUD e dashboard ('n/d' se non ancora fissato)."""
        return "n/d" if self.best_fitness == -math.inf else f"{self.best_fitness:.0f}"

    # ── Dashboard headless: orchestratore ──

    def _draw_stats_screen(self, genome_num, total, config=None):
        """Disegna la dashboard completa per il training headless."""
        self.screen.fill((10, 10, 30))

        # Header
        self._draw_header(config)

        # Colonna sinistra (x=20..620)
        lx = 20
        ly = 80
        ly = self._draw_section_progress(lx, ly, genome_num, total, config)
        ly = self._draw_section_fitness(lx, ly)
        ly = self._draw_section_behavior(lx, ly)
        self._draw_section_efficiency(lx, ly, genome_num, config)

        # Colonna destra (x=650..1260)
        rx = 650
        ry = 80
        ry = self._draw_section_breakdown(rx, ry)
        self._draw_section_species(rx, ry)

        # Footer: barra progresso + hint
        self._draw_footer(genome_num, total)

    # ── Header ──

    def _draw_header(self, config=None):
        pop = config.pop_size if config else "?"
        elapsed = self._format_time(time.time() - self._training_start_time) if self._training_start_time else "0s"

        title = self._font_big.render("YELLOW STAR - TRAINING AI", True, (255, 255, 0))
        self.screen.blit(title, title.get_rect(center=(SCREEN_WIDTH // 2, 22)))

        sub = self._font_small.render(
            f"Gen {self.generation + 1}  |  Pop {pop}  |  {elapsed}",
            True, (150, 150, 200)
        )
        self.screen.blit(sub, sub.get_rect(center=(SCREEN_WIDTH // 2, 50)))

        pygame.draw.line(self.screen, (60, 60, 100), (20, 65), (SCREEN_WIDTH - 20, 65), 1)

    # ── Sezione A: Progresso ──

    def _draw_section_progress(self, x, y, genome_num, total, config):
        y = self._draw_section_header("PROGRESSO", x, y)

        # Generazione + tempo gen sulla stessa riga
        gen_text = f"Gen {self.generation + 1}"
        if self._gen_start_time:
            elapsed = time.time() - self._gen_start_time
            gen_text += f"  |  ~{elapsed:.0f}s"
            if elapsed > 0 and genome_num > 0:
                gen_text += f"  |  {genome_num / elapsed:.1f} g/s"
        self._draw_kv(x, y, "", gen_text, (200, 200, 255))
        y += 20

        # Barra inline per genoma corrente
        progress = genome_num / max(1, total)
        lbl = self._font_small.render(f"Genoma {genome_num}/{total}", True, (180, 180, 200))
        self.screen.blit(lbl, (x, y))
        bar_x = x + 170
        bar_w = 200
        bar_h = 14
        pygame.draw.rect(self.screen, (40, 40, 70), (bar_x, y + 2, bar_w, bar_h))
        pygame.draw.rect(self.screen, (80, 180, 80), (bar_x, y + 2, int(bar_w * progress), bar_h))
        pygame.draw.rect(self.screen, (100, 100, 140), (bar_x, y + 2, bar_w, bar_h), 1)
        pct = self._font_small.render(f"{progress * 100:.0f}%", True, (255, 255, 255))
        self.screen.blit(pct, (bar_x + bar_w + 8, y))
        y += 20

        # Validazione dei migliori a fine generazione (o all'avvio)
        if self._validation_progress:
            done, n = self._validation_progress
            self._draw_kv(x, y, "Validazione:", f"episodio {done}/{n}", (255, 200, 100))
            y += 20

        return y + 6

    # ── Sezione B: Fitness ──

    def _draw_section_fitness(self, x, y):
        y = self._draw_section_header("FITNESS", x, y)

        # Record di validazione (media sulla suite, senza imitation) e
        # migliore validato dell'ultima generazione
        record_color = (100, 255, 100)
        record_text = self._format_record()
        if self._new_record_this_gen:
            record_text += "  *NUOVO*"
        elif self._last_validation:
            record_text += f"  (ultima gen {self._last_validation[1]:.0f})"
        self._draw_kv(x, y, "Record val.:", record_text, record_color)
        y += 20

        # Gen corrente (compatta: avg / best / stdev su 2 righe)
        if self._current_gen_fitnesses:
            n = len(self._current_gen_fitnesses)
            avg = sum(self._current_gen_fitnesses) / n
            best = max(self._current_gen_fitnesses)
            stdev = 0.0
            if n > 1:
                stdev = (sum((f - avg) ** 2 for f in self._current_gen_fitnesses) / (n - 1)) ** 0.5
            self._draw_kv(x, y, "Corrente:", f"avg {avg:.0f}  best {best:.0f}  \u00b1{stdev:.0f}", (180, 220, 180))
            y += 20
        else:
            surf = self._font_small.render("In attesa...", True, (120, 120, 150))
            self.screen.blit(surf, (x, y))
            y += 20

        # Gen precedente (una riga)
        if self._last_gen_stats:
            self._draw_kv(x, y, "Precedente:", f"avg {self._last_gen_stats['avg_fitness']:.0f}  best {self._last_gen_stats['best_fitness']:.0f}", (160, 180, 200))
            y += 20

        # Trend (una riga)
        trend = self._calc_trend(5)
        if trend:
            avg_t, best_t = trend
            self._draw_trend_indicator(x, y, avg_t, "AVG")
            self._draw_trend_indicator(x + 240, y, best_t, "BEST")
            y += 20
        elif len(self._fitness_history) > 0:
            surf = self._font_small.render(
                f"Trend: in calcolo... ({len(self._fitness_history)}/10 gen)",
                True, (120, 120, 150)
            )
            self.screen.blit(surf, (x, y))
            y += 20

        return y + 6

    # ── Sezione C: Scomposizione Fitness ──

    def _draw_section_breakdown(self, x, y):
        y = self._draw_section_header("SCOMPOSIZIONE FITNESS", x, y)

        # Usa il miglior genoma della gen corrente, o il best di sempre
        comp = None
        label_src = ""
        if self._current_gen_fitness_components:
            best_idx = 0
            if self._current_gen_fitnesses:
                best_val = max(self._current_gen_fitnesses)
                best_idx = self._current_gen_fitnesses.index(best_val)
                best_idx = min(best_idx, len(self._current_gen_fitness_components) - 1)
            comp = self._current_gen_fitness_components[best_idx]
            label_src = "best gen corrente"
        elif self._best_genome_components:
            comp = self._best_genome_components
            label_src = "record assoluto"

        if not comp:
            surf = self._font_small.render("In attesa...", True, (120, 120, 150))
            self.screen.blit(surf, (x, y))
            return y + 20

        src = self._font_small.render(f"({label_src})", True, (120, 120, 150))
        self.screen.blit(src, (x, y))
        y += 18

        # Tutti i componenti della fitness (imitation compreso): TOTALE e'
        # la fitness del genoma (per il record: media di validazione)
        items = [
            ("Kills",      comp.get('comp_kills', 0),               (120, 255, 120)),
            ("Kill miss.", comp.get('comp_missile_kills', 0),       (255, 180, 80)),
            ("Kill bomba", comp.get('comp_bomb_kills', 0),          (200, 80, 255)),
            ("Schivate",   comp.get('comp_dodges', 0),              (100, 220, 180)),
            ("Near-miss",  comp.get('comp_near_misses', 0),         (100, 255, 200)),
            ("Precisione", comp.get('comp_accuracy', 0),            (180, 255, 100)),
            ("Sopravv.",   comp['comp_frames'],                     (100, 200, 255)),
            ("Livello",    comp['comp_level'],                      (255, 255, 100)),
            ("Power-up",   comp['comp_powerups'],                   (200, 100, 255)),
            ("PU avvic.",  comp.get('comp_pu_approach', 0),         (180, 100, 255)),
            ("Missili",    comp['comp_missiles'],                   (255, 150, 50)),
            ("Upgrade",    comp['comp_upgrades'],                   (80, 220, 200)),
            ("Vite",       comp['comp_lives'],                      (255, 80, 80)),
            ("Scudo",      comp['comp_shield'],                     (255, 120, 50)),
            ("Idle",       comp['comp_idle'],                       (200, 150, 100)),
            ("Sparo",      comp.get('comp_shoot_penalty', 0),       (180, 160, 120)),
            ("Imitation",  comp.get('comp_imitation', 0),           (255, 200, 200)),
        ]

        max_abs = max(abs(v) for _, v, _ in items) if items else 1
        max_abs = max(max_abs, 1)
        bar_max_w = 140
        row_h = 18

        for label, value, color in items:
            lbl = self._font_small.render(label, True, (180, 180, 200))
            self.screen.blit(lbl, (x, y))

            sign = "+" if value >= 0 else ""
            val_text = f"{sign}{value:.0f}"
            val_surf = self._font_small.render(val_text, True, color)
            self.screen.blit(val_surf, (x + 105, y))

            bar_w = int(abs(value) / max_abs * bar_max_w)
            bar_x = x + 185
            pygame.draw.rect(self.screen, (40, 40, 70), (bar_x, y + 3, bar_max_w, 10))
            if bar_w > 0:
                pygame.draw.rect(self.screen, color, (bar_x, y + 3, bar_w, 10))

            y += row_h

        # Totale
        pygame.draw.line(self.screen, (60, 60, 100), (x, y), (x + 340, y), 1)
        y += 3
        total_fit = sum(v for _, v, _ in items)
        self._draw_kv(x, y, "TOTALE:", f"{total_fit:.0f}", (255, 255, 255))

        return y + 20

    # ── Sezione D: Specie ──

    def _draw_section_species(self, x, y):
        y = self._draw_section_header("SPECIE", x, y)

        species_data = []
        max_stagnation = 0

        if self._population and hasattr(self._population, 'species'):
            species_set = self._population.species
            for sid, sp in species_set.species.items():
                size = len(sp.members)
                fit = sp.fitness if sp.fitness else 0
                stag = self.generation - (sp.last_improved or 0) if hasattr(sp, 'last_improved') and sp.last_improved is not None else 0
                species_data.append((sid, size, fit, stag))
                max_stagnation = max(max_stagnation, stag)

        num_species = len(species_data)
        # Specie + stagnazione su una riga
        stag_color = (255, 100, 100) if max_stagnation >= 10 else (255, 255, 100) if max_stagnation >= 5 else (100, 255, 100)
        self._draw_kv(x, y, f"Attive: {num_species}", f"Stagnazione: {max_stagnation}", stag_color)
        y += 20

        if not species_data:
            return y + 6

        # Ordina per dimensione decrescente, mostra max 5
        species_data.sort(key=lambda s: s[1], reverse=True)
        total_pop = sum(s[1] for s in species_data)
        bar_max_w = 130
        colors_species = [
            (100, 220, 255), (80, 200, 220), (60, 180, 200),
            (50, 160, 180), (40, 140, 160),
        ]

        for i, (sid, size, fit, stag) in enumerate(species_data[:5]):
            bar_color = colors_species[min(i, len(colors_species) - 1)]
            if stag >= 10:
                bar_color = (255, 120, 50)

            bar_w = int(size / max(1, total_pop) * bar_max_w)
            lbl = self._font_small.render(f"S{sid}:", True, (150, 150, 180))
            self.screen.blit(lbl, (x, y))

            bar_x = x + 45
            pygame.draw.rect(self.screen, (40, 40, 70), (bar_x, y + 2, bar_max_w, 12))
            if bar_w > 0:
                pygame.draw.rect(self.screen, bar_color, (bar_x, y + 2, bar_w, 12))

            info = self._font_small.render(f"{size}  fit:{fit:.0f}", True, (180, 180, 200))
            self.screen.blit(info, (bar_x + bar_max_w + 6, y))
            y += 18

        return y + 6

    # ── Sezione E: Efficienza ──

    def _draw_section_efficiency(self, x, y, genome_num, config):
        y = self._draw_section_header("EFFICIENZA", x, y)

        # Tempo totale + gen/min su una riga (generazioni di questa sessione:
        # dopo un resume self.generation parte dal checkpoint)
        if self._training_start_time:
            elapsed = time.time() - self._training_start_time
            eff_text = self._format_time(elapsed)
            gens_done = self.generation - self._start_generation
            if gens_done > 0:
                eff_text += f"  |  {gens_done / (elapsed / 60.0):.2f} gen/min"
            self._draw_kv(x, y, "Tempo:", eff_text, (200, 200, 100))
            y += 20

        # Tempo/gen medio + ultimo su una riga
        if self._gen_times:
            avg_t = sum(self._gen_times) / len(self._gen_times)
            last_t = self._gen_times[-1]
            t_color = (100, 255, 100) if last_t <= avg_t else (255, 200, 100)
            self._draw_kv(x, y, "Tempo/gen:", f"avg {avg_t:.1f}s  last {last_t:.1f}s", t_color)
            y += 20

        # Genomi valutati in questa sessione
        self._draw_kv(x, y, "Genomi valutati:", str(self._genomes_evaluated), (180, 180, 200))
        y += 20

        return y + 6

    # ── Sezione F: Comportamento AI ──

    def _draw_section_behavior(self, x, y):
        y = self._draw_section_header("COMPORTAMENTO AI (medie gen)", x, y)

        if not self._current_gen_scores:
            surf = self._font_small.render("In attesa...", True, (120, 120, 150))
            self.screen.blit(surf, (x, y))
            return y + 20

        n = len(self._current_gen_scores)
        avg_score = sum(self._current_gen_scores) / n
        avg_level = sum(self._current_gen_levels) / n
        avg_frames = sum(self._current_gen_frames) / n
        avg_lives = sum(self._current_gen_lives_lost) / n
        avg_powerups = sum(self._current_gen_powerups) / n
        avg_missiles = sum(self._current_gen_missiles) / n
        avg_upgrades = sum(self._current_gen_upgrades) / n
        avg_idle = sum(self._current_gen_idle_frames) / n
        idle_pct = (avg_idle / max(1, avg_frames)) * 100

        # Riga 1: Score + Livello + Frames
        self._draw_kv(x, y, "Score:", f"~{avg_score:.0f}", (180, 220, 180))
        self._draw_kv(x + 170, y, "Lv:", f"~{avg_level:.1f}", (255, 255, 100))
        self._draw_kv(x + 310, y, "Frame:", f"~{avg_frames:.0f}", (100, 200, 255))
        y += 20

        # Riga 2: Vite + Idle + Power-up
        lives_color = (100, 255, 100) if avg_lives < 1 else (255, 255, 100) if avg_lives < 2 else (255, 100, 100)
        idle_color = (100, 255, 100) if idle_pct < 15 else (255, 255, 100) if idle_pct < 30 else (255, 100, 100)
        self._draw_kv(x, y, "Vite perse:", f"~{avg_lives:.1f}", lives_color)
        self._draw_kv(x + 170, y, "Idle:", f"~{idle_pct:.0f}%", idle_color)
        self._draw_kv(x + 310, y, "PU/Mis:", f"~{avg_powerups:.1f}/{avg_missiles:.1f}", (200, 100, 255))
        y += 20

        return y + 6

    # ── Footer ──

    def _draw_footer(self, genome_num, total):
        y = SCREEN_HEIGHT - 45
        bar_x = 30
        bar_w = SCREEN_WIDTH - 60
        bar_h = 18
        progress = genome_num / max(1, total)

        pygame.draw.rect(self.screen, (40, 40, 70), (bar_x, y, bar_w, bar_h))
        fill_w = int(bar_w * progress)
        if fill_w > 0:
            pygame.draw.rect(self.screen, (80, 180, 80), (bar_x, y, fill_w, bar_h))
        pygame.draw.rect(self.screen, (100, 100, 140), (bar_x, y, bar_w, bar_h), 1)

        pct = self._font_small.render(f"{progress * 100:.0f}%", True, (255, 255, 255))
        self.screen.blit(pct, pct.get_rect(center=(SCREEN_WIDTH // 2, y + bar_h // 2)))

        esc = self._font_small.render("ESC per interrompere", True, (100, 100, 120))
        self.screen.blit(esc, (SCREEN_WIDTH - 200, y + bar_h + 4))

    # ── Helper: intestazione sezione ──

    def _draw_section_header(self, title, x, y, width=560):
        """Disegna un'intestazione di sezione con sottolineatura."""
        surf = self._font_section.render(title, True, (255, 255, 0))
        self.screen.blit(surf, (x, y))
        line_y = y + 20
        pygame.draw.line(self.screen, (60, 60, 100), (x, line_y), (x + width, line_y), 1)
        return y + 26

    # ── Helper: coppia chiave-valore ──

    def _draw_kv(self, x, y, label, value, color=(200, 200, 255)):
        """Disegna label (grigio) + valore (colore) sulla stessa riga."""
        lbl = self._font_small.render(label, True, (180, 180, 200))
        val = self._font_small.render(str(value), True, color)
        self.screen.blit(lbl, (x, y))
        self.screen.blit(val, (x + lbl.get_width() + 8, y))

    # ── Helper: indicatore trend ──

    def _draw_trend_indicator(self, x, y, value, label):
        """Disegna un valore di trend con freccia e colore."""
        if value > 2:
            color = (100, 255, 100)
            arrow = "\u2191"  # ↑
        elif value < -2:
            color = (255, 100, 100)
            arrow = "\u2193"  # ↓
        else:
            color = (255, 255, 100)
            arrow = "~"

        text = f"{arrow} {value:+.1f}% {label}"
        surf = self._font_small.render(text, True, color)
        self.screen.blit(surf, (x, y))

    # ── Helper: calcolo trend ──

    def _calc_trend(self, n=5):
        """Confronta le ultime n generazioni con le n precedenti.

        Returns (avg_trend_pct, best_trend_pct) o None se dati insufficienti.
        """
        if len(self._fitness_history) < n * 2:
            return None
        recent = self._fitness_history[-n:]
        previous = self._fitness_history[-2 * n:-n]
        recent_avg = sum(r[0] for r in recent) / n
        prev_avg = sum(r[0] for r in previous) / n
        recent_best = sum(r[1] for r in recent) / n
        prev_best = sum(r[1] for r in previous) / n

        avg_delta = ((recent_avg - prev_avg) / max(1, abs(prev_avg))) * 100
        best_delta = ((recent_best - prev_best) / max(1, abs(prev_best))) * 100
        return (avg_delta, best_delta)

    # ── Helper: formattazione tempo ──

    @staticmethod
    def _format_time(seconds):
        """Formatta secondi in stringa leggibile (es. '1h 23m', '45s')."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        minutes = seconds / 60
        if minutes < 60:
            return f"{minutes:.0f}m {seconds % 60:.0f}s"
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        return f"{hours}h {mins}m"

    def _imitation_bonus(self, genome, config):
        """Bonus fitness per somiglianza alle azioni umane registrate.

        Calcolato nel main solo dal training visuale: in headless lo calcola
        il worker (_eval_genome_worker) con gli stessi campioni e frequenze.
        """
        if self._imitation_samples is None:
            return 0.0
        return _imitation_bonus_from_samples(
            compile_network(genome, config), self._imitation_samples,
            self._imitation_human_freq,
        )

    @staticmethod
    def _action_similarity(ai_action, human_action):
        """Calcola somiglianza tra azione AI e azione umana (0-1).

        Media dell'accordo su movimento verticale, orizzontale (match esatto
        = 1, off-by-one = 0.5) e sparo (match esatto); missile e bomba sono
        esclusi (vedi _IMITATION_COMPONENTS).
        """
        score = 0.0
        for c in _IMITATION_COMPONENTS:
            score += _component_agreement(c, ai_action[c], human_action[c])
        return score / len(_IMITATION_COMPONENTS)

    @staticmethod
    def _decode_actions(output):
        """Converte output della rete in azioni di gioco."""
        return _decode_actions_static(output)

    def _load_imitation_samples(self):
        """Carica le sessioni umane ed estrae i campioni per l'imitation bonus.

        Usa le IMITATION_TOP_SESSIONS sessioni valide con lo score piu' alto
        (a parita' la piu' recente), ognuna pesata
        min(IMITATION_WEIGHT_CAP, ln(1 + score / 1000)): il tetto impedisce
        che una sola partita domini il bonus. I file si leggono uno alla volta
        e di ogni sessione si tengono solo IMITATION_SAMPLES_PER_SESSION frame
        equidistanti (le sessioni complete occupano decine di MB in RAM).
        File illeggibili o sessioni malformate (chiavi mancanti, obs di
        lunghezza diversa da num_inputs) vengono saltati con un avviso.

        Imposta _imitation_samples, lista di (obs, azione umana, peso) o None
        se non c'e' nessuna sessione valida, _imitation_human_freq (frequenze
        pesate delle azioni umane, per la somiglianza attesa per caso) e
        _human_sessions_count (file .pkl su disco, mostrato nell'overlay del
        training visuale).
        """
        self._imitation_samples = None
        self._imitation_human_freq = None
        self._human_sessions_count = 0
        if not os.path.isdir(SESSIONS_DIR):
            return

        filenames = sorted(f for f in os.listdir(SESSIONS_DIR) if f.endswith('.pkl'))
        self._human_sessions_count = len(filenames)
        num_inputs = self.config.genome_config.num_inputs
        candidates = []  # (score, nome file, [(obs, azione)])
        for filename in filenames:
            filepath = os.path.join(SESSIONS_DIR, filename)
            try:
                with open(filepath, 'rb') as f:
                    session = pickle.load(f)
                score = float(session['score'])
                frames = session['frames']
                n = min(IMITATION_SAMPLES_PER_SESSION, len(frames))
                picked = []
                for i in range(n):
                    obs, action = frames[i * len(frames) // n]
                    if len(obs) != num_inputs:
                        raise ValueError(f"obs di {len(obs)} valori invece di {num_inputs}")
                    action = {c: action[c] for c in _IMITATION_COMPONENTS}
                    if any(action[c] not in values for c, values in _IMITATION_COMPONENTS.items()):
                        raise ValueError(f"azione non valida: {action}")
                    picked.append((obs, action))
            except Exception as e:
                # pickle e sessioni malformate possono sollevare quasi ogni
                # eccezione (UnpicklingError, EOFError, AttributeError,
                # ModuleNotFoundError, KeyError, TypeError, ...)
                print(f"Sessione umana ignorata ({filename}): {e!r}")
                continue
            if picked:
                candidates.append((score, filename, picked))

        if not candidates:
            return
        candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
        samples = []
        for score, _filename, picked in candidates[:IMITATION_TOP_SESSIONS]:
            weight = min(IMITATION_WEIGHT_CAP, math.log1p(max(0.0, score) / 1000.0))
            samples.extend((obs, action, weight) for obs, action in picked)
        self._imitation_samples = samples
        self._imitation_human_freq = _action_frequencies(samples)

    def _find_latest_checkpoint(self):
        """Trova il checkpoint NEAT leggibile con la generazione piu' alta.

        Ignora i suffissi non numerici e salta i file illeggibili (troncati
        o corrotti), passando al precedente. Ritorna (path, config salvata
        nel checkpoint) oppure (None, None).
        """
        for _generation, path in reversed(_list_checkpoints()):
            try:
                with gzip.open(path) as f:
                    data = pickle.load(f)
                return path, data[1]
            except Exception as e:
                print(f"Checkpoint illeggibile, ignorato: {path} ({e!r})")
        return None, None

class _TrainingInterrupted(Exception):
    """Eccezione interna per interrompere il training con ESC."""
    pass
