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
import itertools
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
from src.game_env import GameEnv, OBS_VERSION
from src.training_dashboard import TrainingDashboard

AI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ai")
CONFIG_PATH = os.path.join(AI_DIR, "neat_config.txt")
BEST_GENOME_PATH = os.path.join(AI_DIR, "best_genome.pkl")
# Record di validazione di best_genome.pkl (media, seed, versione della fitness)
BEST_GENOME_META_PATH = os.path.join(AI_DIR, "best_genome_meta.json")
CHECKPOINT_PREFIX = os.path.join(AI_DIR, "neat-checkpoint-")
MAX_CHECKPOINTS = 2  # mantieni solo gli ultimi N checkpoint
SESSIONS_DIR = os.path.join(AI_DIR, "human_sessions")
# Storico per generazione (stato del training, versionato) e cartella della
# dashboard HTML (index.html versionato, data.js e live.js generati)
HISTORY_PATH = os.path.join(AI_DIR, "training_history.jsonl")
DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard")
DASHBOARD_AUTO_OPEN = True  # apre la dashboard nel browser all'avvio del training

# Durate in secondi di gioco, convertite in frame con settings.FPS (GameEnv
# avanza di 1000/FPS ms a ogni step)
# 5 minuti (180 s fino alla fitness v6: con l'osservazione v2 i migliori
# arrivavano a 165-171 s e dalla gen 50 la popolazione non migliorava piu')
MAX_EPISODE_SECONDS = 300
MIN_EPISODE_SECONDS = 30   # alla generazione 0, poi cresce fino a MAX in 50 gen
MAX_FRAMES_PER_EPISODE = int(MAX_EPISODE_SECONDS * FPS)  # 18000 a 60 FPS
MIN_FRAMES_PER_EPISODE = int(MIN_EPISODE_SECONDS * FPS)  # 1800 a 60 FPS

# Velocita' di rendering durante il training visuale
TRAIN_RENDER_SKIP = 4  # mostra 1 frame su 4

# Popolazione del training headless (il visuale usa pop_size di neat_config.txt)
HEADLESS_POP_SIZE = 200

# Episodi per genoma: stessi seed per tutti i genomi della generazione
# (common random numbers), nuovi a ogni generazione; fitness = media.
# Con 2 la classifica dei migliori era dominata dal caso (ICC 0.58)
EPISODES_PER_GENOME = 4

# Speciazione: numero di specie che il threshold adattivo cerca di mantenere
# (_check_and_fix_stagnation). Il threshold si calibra sulle distanze della
# generazione: con un passo fisso del 5% il numero di specie oscillava tra 3
# e 12 (le specie in eccesso spariscono solo per stagnazione, in ritardo)
SPECIES_TARGET_MIN = 5
SPECIES_TARGET_MAX = 10
THRESHOLD_MIN = 0.3
THRESHOLD_MAX = 8.0
THRESHOLD_MAX_DROP = 0.15  # discesa massima per generazione (la salita e' libera)

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
FITNESS_VERSION = 7

# Validazione del best in due fasi, su episodi interi (MAX_FRAMES_PER_EPISODE,
# niente early stop, niente imitation) con seed fissi fuori dalla serie del
# training:
# 1. selezione: i VALIDATION_TOP_K genomi con la fitness di gioco piu' alta
#    giocano i VALIDATION_SEEDS;
# 2. conferma: il migliore della selezione, se la sua media supera il record,
#    gioca i CONFIRMATION_SEEDS; best_genome.pkl e' il genoma con la media di
#    conferma piu' alta. Il record e' misurato su seed che non hanno scelto il
#    candidato: con una sola suite da 10 seed il record era il massimo di
#    migliaia di medie rumorose sulle stesse partite (2779 sui 10 seed, 1454
#    su 30 seed nuovi)
VALIDATION_TOP_K = 5
VALIDATION_SEEDS = tuple(range(10 ** 9, 10 ** 9 + 10))
CONFIRMATION_SEEDS = tuple(range(10 ** 9 + 1000, 10 ** 9 + 1040))


# Peso dei power-up raccolti e del saldo di avvicinamento (fitness v5, prima
# 50 e 3): lo stesso genoma con arma, scudo e motore al livello 6 fa 10058
# punti e 122 s invece di 2925 e 52 s (~400-650 punti di partita per livello),
# ma il premio di 70 per raccolta non si vedeva in episodi di ~50 s e l'AI
# raccoglieva ~1 power-up a partita contro i 30 dei giocatori umani
POWERUP_WEIGHT = 250
PU_APPROACH_WEIGHT = 10

# Componenti di _compute_fitness come li spiega la dashboard: tenerli allineati
# ai pesi qui sotto. (chiave, nome, peso, descrizione)
FITNESS_COMPONENTS = (
    ('comp_kills', "Uccisioni con i proiettili", "+50 per nemico",
     "Nemici distrutti dai proiettili."),
    ('comp_missile_kills', "Uccisioni con i missili", "+50 per nemico",
     "Nemici distrutti dai missili a ricerca."),
    ('comp_bomb_kills', "Uccisioni con la bomba", "+50 per nemico",
     "Nemici cancellati da una bomba."),
    ('comp_accuracy', "Precisione", "+10 × precisione × uccisioni",
     "Quota di raffiche andate a segno, moltiplicata per le uccisioni con i proiettili."),
    ('comp_frames', "Sopravvivenza", "+1,5 al secondo",
     "Secondi di gioco prima della fine dell'episodio."),
    ('comp_lives', "Vite perse", "−200 per vita",
     "Ogni vita persa."),
    ('comp_level', "Livello raggiunto", "+50 × livello²",
     "Il livello sale con lo score: premia la progressione nel gioco."),
    ('comp_powerups', "Potenziamenti raccolti", f"+{POWERUP_WEIGHT} ciascuno",
     "Power-up di arma, motore o scudo raccolti."),
    ('comp_missiles', "Missili raccolti", "+30 ciascuno",
     "Power-up dei missili raccolti."),
    ('comp_upgrades', "Livelli di potenziamento", "+20 per livello",
     "Somma dei livelli di arma, motore e scudo raggiunti."),
    ('comp_shield', "Colpi allo scudo", "−15 per colpo, −120 se si rompe",
     "Colpi assorbiti dallo scudo e scudi distrutti."),
    ('comp_idle', "Inattività", "−15 al secondo",
     "Secondi in cui la nave non si sposta, anche se spinge contro un bordo."),
    ('comp_dodges', "Schivate", "+5 ciascuna",
     "Nemici o asteroidi passati entro 60 px e usciti senza colpire."),
    ('comp_near_misses', "Passaggi ravvicinati", "+3 ciascuno",
     "Minacce arrivate entro 60 px dalla nave."),
    ('comp_pu_approach', "Avvicinamento ai power-up", f"+{PU_APPROACH_WEIGHT} al secondo netto",
     "Saldo dei secondi passati ad avvicinarsi al power-up più vicino (mai negativo)."),
    ('comp_shoot_penalty', "Raffiche sparate", "−0,01 per raffica",
     "Piccolo costo per ogni raffica, contro lo sparo continuo inutile."),
    ('comp_imitation', "Imitazione del giocatore", f"fino a +{IMITATION_MAX_BONUS}",
     f"Somiglianza con le partite umane migliori oltre il caso (dalla generazione {IMITATION_START_GEN})."),
)


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
        + power-up x POWERUP_WEIGHT (250) + missili raccolti x 30 + livelli upgrade x 20
        - colpi allo scudo x 15 - scudi rotti x 120
        - secondi fermo x 15 + near-miss x 3
        + saldo netto (>= 0) dei secondi in avvicinamento a un power-up x PU_APPROACH_WEIGHT (10)
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
        'comp_powerups': env.powerups_collected * POWERUP_WEIGHT,
        'comp_missiles': env.missiles_collected * 30,
        'comp_upgrades': env.total_upgrade_levels * 20,
        'comp_shield': -(env.shield_hits * 15 + env.shield_breaks * 120),
        'comp_idle': -(env.idle_frames / FPS * 15),
        'comp_near_misses': env.near_misses * 3,
        'comp_pu_approach': max(0, env.powerup_approach_frames) / FPS * PU_APPROACH_WEIGHT,
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
        'missiles_fired': env.missiles_fired,
        'bombs_used': env.bombs_used,
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
        'missiles_fired': 0, 'bombs_used': 0,
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


def _game_stats(s):
    """Statistiche di gioco leggibili da un dict env_stats (anche medio)."""
    frames = s.get('frame_count', 0)
    return {
        'score': s.get('score', 0),
        'surv': frames / FPS,
        'level': s.get('level', 1),
        'lives_lost': s.get('lives_lost', 0),
        'kills': s.get('enemies_killed', 0),
        'missile_kills': s.get('missile_kills', 0),
        'bomb_kills': s.get('bomb_kills', 0),
        'missiles_fired': s.get('missiles_fired', 0),
        'missiles': s.get('missiles', 0),
        'bombs_used': s.get('bombs_used', 0),
        'powerups': s.get('powerups', 0),
        'upgrades': s.get('upgrades', 0),
        'idle_pct': s.get('idle_frames', 0) / frames * 100 if frames else 0.0,
        'accuracy': s.get('accuracy', 0) * 100,
        'near_misses': s.get('near_misses', 0),
        'dodges': s.get('dodges', 0),
    }


def _mean_dicts(dicts):
    """Media chiave per chiave di dict con le stesse chiavi."""
    n = len(dicts)
    return {k: sum(d[k] for d in dicts) / n for k in dicts[0]} if n else {}


def _summary(values):
    """Media, mediana, percentili 10/90, minimo, massimo e deviazione standard."""
    v = sorted(values)
    n = len(v)
    if not n:
        return None
    mean = sum(v) / n

    def q(p):
        i = (n - 1) * p
        lo = int(i)
        hi = min(lo + 1, n - 1)
        return v[lo] + (v[hi] - v[lo]) * (i - lo)

    sd = (sum((x - mean) ** 2 for x in v) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return {'mean': mean, 'median': q(0.5), 'p10': q(0.1), 'p90': q(0.9),
            'min': v[0], 'max': v[-1], 'sd': sd}


def _trend(values, k=15):
    """Confronta le ultime k generazioni con le k precedenti.

    Ritorna (z, variazione %) o None con meno di 8 valori; |z| >= 2 indica
    una differenza oltre il rumore tra generazioni.
    """
    v = [x for x in values if x is not None]
    if len(v) < 8:
        return None
    k = min(k, len(v) // 2)
    a, b = v[-2 * k:-k], v[-k:]
    ma, mb = sum(a) / k, sum(b) / k
    va = sum((x - ma) ** 2 for x in a) / (k - 1)
    vb = sum((x - mb) ** 2 for x in b) / (k - 1)
    se = ((va + vb) / k) ** 0.5
    z = (mb - ma) / se if se > 0 else 0.0
    pct = (mb - ma) / abs(ma) * 100 if ma else 0.0
    return z, pct


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


def _prune_unexpressed(genome, config):
    """Elimina i geni che non entrano nella rete espressa (fenotipo invariato).

    Tiene le uscite, i nodi valutati da FeedForwardNetwork.create e le sole
    connessioni che quei nodi leggono: spariscono le connessioni disabilitate,
    i nodi senza percorso verso un'uscita e i loro collegamenti. La rete
    compilata resta identica bit per bit (stessi nodi, stessi link nello
    stesso ordine). Senza potatura le reti accumulavano ~90 nodi e ~440
    connessioni con ~13 espresse: mutate_add_connection sceglie la
    destinazione tra tutti i nodi, quindi quasi ogni mutazione strutturale
    finiva su geni morti, e la popolazione usava in mediana 6 input su 52.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    keep_nodes = set(config.genome_config.output_keys)
    keep_conns = set()
    for node, _act, _agg, _bias, _resp, links in net.node_evals:
        keep_nodes.add(node)
        keep_conns.update((i, node) for i, _w in links)
    for key in [k for k in genome.connections if k not in keep_conns]:
        del genome.connections[key]
    for key in [k for k in genome.nodes if k not in keep_nodes]:
        del genome.nodes[key]


def _distance_matrix(genomes, genome_config):
    """Distanze genetiche (DefaultGenome.distance) tra tutte le coppie."""
    n = len(genomes)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        gi, row = genomes[i], dist[i]
        for j in range(i + 1, n):
            row[j] = dist[j][i] = gi.distance(genomes[j], genome_config)
    return dist


def _greedy_species_count(dist, threshold):
    """Specie che si formerebbero da zero con `threshold`.

    Stesso criterio di DefaultSpeciesSet: un genoma entra nella specie del
    primo rappresentante a distanza < threshold, altrimenti ne fonda una.
    """
    reps = []
    for i, row in enumerate(dist):
        if all(row[r] >= threshold for r in reps):
            reps.append(i)
    return len(reps)


def _threshold_for_species(dist, target):
    """Threshold piu' basso che forma al massimo `target` specie (bisezione)."""
    lo, hi = THRESHOLD_MIN, THRESHOLD_MAX
    if _greedy_species_count(dist, lo) <= target:
        return lo
    if _greedy_species_count(dist, hi) > target:
        return hi
    for _ in range(20):
        mid = (lo * hi) ** 0.5
        if _greedy_species_count(dist, mid) <= target:
            hi = mid
        else:
            lo = mid
    return hi


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
        # Record di validazione: media della suite di conferma
        # (CONFIRMATION_SEEDS) del genoma in best_genome.pkl, -inf se non
        # ancora stabilito
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
        self._fonts = None
        # Riferimenti NEAT (set in train())
        self._population = None
        self._stats_reporter = None
        self._checkpointer = None
        # Dashboard HTML (storico, dettaglio dell'ultima generazione, stato live)
        self._dash = None
        self._training_start_time = None
        self._gen_start_time = None
        self._gen_total = 0             # genomi della generazione corrente
        self._gen_rows = []             # genomi valutati: genoma, fitness, env_stats
        self._progress = ('avvio', 0, 0)  # fase, completati, totale
        self._new_record_this_gen = False
        # Record: chiave, valore, statistiche e fitness per seed del genoma in
        # best_genome.pkl; record a inizio sessione per il delta di sessione
        self._record_info = None
        self._session_start_record = -math.inf
        # Validazione: cache per genome.key (media, statistiche medie, fitness
        # per seed) della selezione e della conferma, valide per tutta la
        # sessione (suite e fitness non cambiano); migliore della selezione e
        # conferma dell'ultima generazione (key, media), None se non giocata
        self._val_cache = {}
        self._conf_cache = {}
        self._last_validation = None
        self._last_confirmation = None
        self._pool = None
        # Fitness accumulata per genome.key dei genomi della generazione
        # corrente: (somma fitness, somma fitness di gioco, generazioni giocate)
        self._fit_hist = {}

    def _init_fonts(self):
        if self._font is None:
            # Overlay del training visuale
            self._font = pygame.font.Font(None, 32)
            self._font_small = pygame.font.Font(None, 24)
            # Schermata del training headless
            name = 'segoeui,arial'
            self._fonts = {
                'title': pygame.font.SysFont(name, 34, bold=True),
                'kpi': pygame.font.SysFont(name, 38, bold=True),
                'body': pygame.font.SysFont(name, 20),
                'label': pygame.font.SysFont(name, 17, bold=True),
                'small': pygame.font.SysFont(name, 16),
            }

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
        # Versioni di fitness e osservazione, salvate nei checkpoint con la config
        self.config.fitness_version = FITNESS_VERSION
        self.config.obs_version = OBS_VERSION

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
            self._restore_node_indexer(p, saved_config)
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

        # Dashboard HTML: storico (anche delle sessioni precedenti) e stato live
        self._dash = TrainingDashboard(self._dashboard_static(p.config),
                                       HISTORY_PATH, DASHBOARD_DIR)
        self._dash.publish()
        self._set_progress('avvio', force=True)
        if DASHBOARD_AUTO_OPEN:
            self._dash.open()

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
            self._session_start_record = self.best_fitness
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
            self._set_progress('fermo', force=True)

        # Il miglior genoma e' gia' salvato incrementalmente in _eval_genomes
        if self.best_genome and self.best_fitness > -math.inf:
            print(f"\nRecord di validazione: {self.best_fitness:.1f}")

    @staticmethod
    def _restore_node_indexer(p, saved_config):
        """Riprende il contatore delle chiavi dei nodi dopo il restore.

        restore_checkpoint con new_config trasferisce il tracker delle
        innovazioni ma non il node_indexer della config pickled: neat lo
        ricreerebbe dal massimo dei nodi del primo genoma mutato e, con le
        reti potate (_prune_unexpressed), riuserebbe chiavi di nodi presenti
        in altri genomi (AssertionError in get_new_node_key). Riparte dal
        massimo tra il contatore salvato e le chiavi della popolazione e dei
        rappresentanti delle specie.
        """
        gc = p.config.genome_config
        genomes = list(p.population.values())
        genomes += [s.representative for s in p.species.species.values() if s.representative]
        next_id = max((k for g in genomes for k in g.nodes), default=gc.num_outputs - 1) + 1
        saved = getattr(saved_config.genome_config, 'node_indexer', None)
        if saved is not None:
            next_id = max(next_id, next(saved))
        gc.node_indexer = itertools.count(next_id)

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
        self._gen_rows = []
        render = self.screen is not None and not self._headless
        game_fitness = {}  # idx -> fitness di gioco (senza imitation)

        max_frames = self._get_max_frames()
        seeds = _generation_seeds(self.generation)
        self._set_progress('valutazione', 0, total, force=True)

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
                self._set_progress('valutazione', completed, total)
                # Aggiorna la schermata durante la valutazione
                if self.screen and (completed % 5 == 0 or completed == total):
                    self._draw_stats_screen()
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
                self._set_progress('valutazione', idx + 1, total)

        # Fitness media su tutte le generazioni giocate (elite riportate)
        self._average_fitness(genomes, game_fitness)

        # Potatura dei geni inespressi: la rete di ogni genoma non cambia, le
        # mutazioni della generazione dopo agiscono sulla rete vera
        for _genome_id, genome in genomes:
            _prune_unexpressed(genome, config)

        # Validazione dei migliori: salva best_genome.pkl se batte il record
        self._new_record_this_gen = self._validate_top_genomes(genomes, game_fitness, config)

        # Storico e dettaglio della generazione per la dashboard
        gen_time = time.time() - self._gen_start_time
        if self._dash is not None and self._gen_rows:
            record, snapshot = self._generation_report(gen_time, max_frames, seeds)
            self._dash.record_generation(record, snapshot)

        self.generation += 1

        # Controllo speciazione (threshold adattivo)
        self._check_and_fix_stagnation(config, [g for _, g in genomes])

        # In headless: aggiorna schermata fine generazione
        if self._headless and self.screen:
            self._draw_stats_screen()
            pygame.display.flip()

    def _average_fitness(self, genomes, game_fitness):
        """Fitness = media di tutte le generazioni giocate dal genoma.

        Le elite passano identiche (stesso oggetto, stessa key) alla
        generazione dopo e rigiocano i nuovi seed: la loro fitness diventa la
        media di tutti gli episodi giocati, quella dei genomi nuovi resta la
        media dei loro EPISODES_PER_GENOME. Un singolo link utile vale ~+600
        di fitness, ma la differenza tra due reti cambia di ~1100 da una
        partita all'altra anche sugli stessi seed: con 4 episodi elite e
        genitori erano scelti in buona parte dal caso. Aggiorna anche
        game_fitness (classifica della validazione) e le righe della
        dashboard; lo storico tiene solo i genomi della generazione.
        """
        hist = {}
        for idx, (_genome_id, genome) in enumerate(genomes):
            total, game, n = self._fit_hist.get(genome.key, (0.0, 0.0, 0))
            total += genome.fitness
            game += game_fitness[idx]
            n += 1
            hist[genome.key] = (total, game, n)
            genome.fitness = total / n
            game_fitness[idx] = game / n
        self._fit_hist = hist
        for row in self._gen_rows:
            row['fitness'] = row['genome'].fitness

    def _poll_interrupt(self):
        """Svuota la coda eventi: ESC o chiusura finestra interrompono il training."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise _TrainingInterrupted()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                raise _TrainingInterrupted()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                self._open_dashboard()

    # ── Validazione e persistenza del best ──

    def _validate_top_genomes(self, genomes, game_fitness, config):
        """Validazione in due fasi dei genomi con la fitness di gioco piu' alta.

        Selezione: i VALIDATION_TOP_K migliori giocano i VALIDATION_SEEDS.
        Conferma: il migliore della selezione, se la sua media supera il
        record, gioca i CONFIRMATION_SEEDS e viene salvato se la media di
        conferma batte il record. I genomi gia' giocati in questa sessione
        (elite riportate identiche alla generazione dopo) escono dalle cache.
        Ritorna True se nuovo record.
        """
        self._last_confirmation = None
        if not game_fitness:
            return False
        ranked = sorted(game_fitness, key=lambda i: game_fitness[i], reverse=True)
        candidates = [genomes[i][1] for i in ranked[:VALIDATION_TOP_K]]
        todo = [g for g in candidates if g.key not in self._val_cache]
        if todo:
            self._val_cache.update(self._run_validation(todo, config))
        best = max(candidates, key=lambda g: self._val_cache[g.key][0])
        screen_mean = self._val_cache[best.key][0]
        self._last_validation = (best.key, screen_mean)
        print(f"  [Validazione] top-{len(candidates)} ({len(todo)} nuovi): migliore genoma "
              f"{best.key}, media {screen_mean:.1f} su {len(VALIDATION_SEEDS)} seed "
              f"(record {self.best_fitness:.1f})")
        if screen_mean <= self.best_fitness:
            return False

        cached = best.key in self._conf_cache
        if not cached:
            self._conf_cache.update(self._run_validation(
                [best], config, phase='conferma', seeds=CONFIRMATION_SEEDS))
        mean, stats, per_seed = self._conf_cache[best.key]
        self._last_confirmation = (best.key, mean)
        print(f"  [Conferma] genoma {best.key}{' (già giocata)' if cached else ''}: "
              f"media {mean:.1f} su {len(CONFIRMATION_SEEDS)} seed (record {self.best_fitness:.1f})")
        if mean <= self.best_fitness:
            return False
        self.best_fitness = mean
        self.best_genome = best
        self._record_info = self._make_record_info(best, mean, stats, per_seed, self.generation)
        self._save_best_genome(best, mean, per_seed, stats)
        return True

    def _run_validation(self, genomes, config, phase='validazione', seeds=VALIDATION_SEEDS):
        """Gioca una suite di validazione (`seeds`) per ogni genoma.

        In headless usa il Pool (un task per genoma e seed), nel visuale gira
        nel processo principale senza rendering. Ritorna {key: (media,
        env_stats medi, [fitness per seed in ordine di seed])}; media -inf se
        un episodio fallisce (genoma incompatibile). `phase` e' la fase
        mostrata da schermata e dashboard durante la suite.
        """
        max_frames = MAX_FRAMES_PER_EPISODE
        tasks = [(g.key, g, config, seed, max_frames) for g in genomes for seed in seeds]
        per_key = {g.key: {} for g in genomes}
        if self._pool is not None:
            results = self._pool.imap_unordered(_validation_worker, tasks, chunksize=1)
        else:
            results = (_validation_worker(t) for t in tasks)
        done = 0
        self._set_progress(phase, 0, len(tasks), force=True)
        for key, seed, fitness, env_stats in results:
            per_key[key][seed] = (fitness, env_stats)
            done += 1
            self._set_progress(phase, done, len(tasks))
            if self.screen is not None:
                if self._headless:
                    self._draw_stats_screen()
                else:
                    self._draw_validation_overlay(done, len(tasks))
                pygame.display.flip()
                self._poll_interrupt()

        out = {}
        for key, by_seed in per_key.items():
            # In ordine di seed: la media non dipende dall'ordine di arrivo
            ordered = [by_seed[s] for s in seeds]
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
        primo genoma confermato viene salvato. La suite del record e' quella di
        conferma (CONFIRMATION_SEEDS).
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
        obs_version = getattr(self.best_genome, 'obs_version', 1)
        if obs_version != OBS_VERSION:
            # Rivalutarlo darebbe un record senza senso: legge valori diversi
            print(f"best_genome.pkl e' per l'osservazione v{obs_version}, attuale "
                  f"v{OBS_VERSION}: record azzerato, il primo genoma confermato lo sostituisce")
            self.best_genome = None
            return

        meta = self._read_best_meta()
        if meta is not None:
            self.best_fitness = float(meta['validation_mean'])
            self._record_info = self._make_record_info(
                self.best_genome, self.best_fitness, meta.get('validation_stats'),
                meta.get('validation_fitness'), meta.get('generation'))
            print(f"Record di validazione: {self.best_fitness:.1f} ({BEST_GENOME_META_PATH})")
            return

        print("Rivalutazione di best_genome.pkl sulla suite di conferma "
              f"({len(CONFIRMATION_SEEDS)} seed, fitness v{FITNESS_VERSION})...")
        mean, stats, per_seed = self._run_validation(
            [self.best_genome], config, phase='rivalutazione',
            seeds=CONFIRMATION_SEEDS)[self.best_genome.key]
        if mean == -math.inf:
            print("  best_genome.pkl non valutabile: record azzerato")
            return
        self.best_fitness = mean
        self._record_info = self._make_record_info(self.best_genome, mean, stats, per_seed, None)
        print(f"  Record di validazione: {mean:.1f}")
        self._write_best_meta(self.best_genome, mean, per_seed, stats)

    def _read_best_meta(self):
        """Sidecar del best se valido per il genoma e la suite correnti, altrimenti None."""
        try:
            with open(BEST_GENOME_META_PATH, encoding='utf-8') as f:
                meta = json.load(f)
            float(meta['validation_mean'])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if (meta.get('fitness_version') != FITNESS_VERSION
                or meta.get('validation_seeds') != list(CONFIRMATION_SEEDS)
                or meta.get('max_frames') != MAX_FRAMES_PER_EPISODE
                or meta.get('genome_key') != getattr(self.best_genome, 'key', None)):
            return None
        return meta

    def _write_best_meta(self, genome, mean, per_seed, stats=None):
        """Scrive il sidecar del best (JSON, atomico: .tmp + os.replace).

        `stats` (env_stats medi della suite) servono alla dashboard per la
        scomposizione della fitness del record anche dopo un riavvio.
        """
        meta = {
            'fitness_version': FITNESS_VERSION,
            'validation_mean': mean,
            'validation_seeds': list(CONFIRMATION_SEEDS),
            'validation_fitness': per_seed,
            'max_frames': MAX_FRAMES_PER_EPISODE,
            'genome_key': genome.key,
            'generation': self.generation,
            'saved_at': datetime.datetime.now().isoformat(timespec='seconds'),
        }
        if stats:
            meta['validation_stats'] = {k: v for k, v in stats.items()
                                        if isinstance(v, (int, float))}
        os.makedirs(os.path.dirname(BEST_GENOME_META_PATH), exist_ok=True)
        tmp = BEST_GENOME_META_PATH + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=1)
        os.replace(tmp, BEST_GENOME_META_PATH)

    def _save_best_genome(self, genome, mean, per_seed, stats=None):
        """Salva best_genome.pkl (stesso formato: il genoma pickled) e il sidecar."""
        os.makedirs(AI_DIR, exist_ok=True)
        # Layout dell'osservazione: la coop e la ripresa rifiutano i genomi
        # di un'altra versione (girerebbero, ma leggendo altri valori)
        genome.obs_version = OBS_VERSION
        tmp = BEST_GENOME_PATH + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(genome, f)
        os.replace(tmp, BEST_GENOME_PATH)
        self._write_best_meta(genome, mean, per_seed, stats)

    def _get_max_frames(self):
        """Frame massimi adattivi: piu' generazioni = piu' tempo concesso."""
        # Cresce da MIN a MAX (30 s -> 5 min di gioco) in 50 generazioni
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
        self._gen_rows.append({
            'genome': genome,
            'fitness': genome.fitness,
            'stats': dict(env_stats, comp_imitation=bonus),
        })

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
                            if event.key == pygame.K_d:
                                self._open_dashboard()

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

    def _check_and_fix_stagnation(self, config, genomes):
        """Controllo della speciazione: adatta il compatibility threshold.

        Agisce sulla speciazione della generazione successiva perche' lo
        species set usa config.species_set_config (anche dopo il restore, vedi
        train()). La stagnazione delle specie la gestisce neat
        (DefaultStagnation: max_stagnation, species_elitism).

        Tiene le specie tra SPECIES_TARGET_MIN e SPECIES_TARGET_MAX (con 20
        specie da ~10 genomi le linee migliori si estinguevano in ~20
        generazioni). Il threshold si calibra sulle distanze di `genomes`
        (la generazione appena valutata): i valori che da zero formerebbero
        SPECIES_TARGET_MAX, il punto medio e SPECIES_TARGET_MIN specie. Con
        poche specie scende al valore del punto medio, con troppe sale al
        valore del punto medio (le specie esistenti restano e spariscono solo
        per stagnazione), altrimenti resta nella fascia tra gli estremi. Un
        passo fisso reagiva al numero di specie, che cambia in ritardo, e
        oscillava tra 3 e 12 specie.
        """
        if not self._population or len(genomes) < 2:
            return

        num_species = len(self._population.species.species)
        threshold = config.species_set_config.compatibility_threshold
        dist = _distance_matrix(genomes, config.genome_config)
        t_lo = _threshold_for_species(dist, SPECIES_TARGET_MAX)
        t_hi = max(t_lo, _threshold_for_species(dist, SPECIES_TARGET_MIN))
        t_mid = min(max(_threshold_for_species(
            dist, (SPECIES_TARGET_MIN + SPECIES_TARGET_MAX) // 2), t_lo), t_hi)
        # Scende al massimo di THRESHOLD_MAX_DROP per generazione: le distanze
        # dei figli mutati sono piu' grandi di quelle dei genitori su cui si
        # calibra, e una specie in piu' vive almeno max_stagnation generazioni
        # (all'avvio da zero, con distanze tutte simili, un salto da 3.0 a
        # 0.56 aveva creato 36 specie su 60 genomi)
        floor = threshold * (1 - THRESHOLD_MAX_DROP)
        if num_species < SPECIES_TARGET_MIN:
            new_t = max(min(threshold, t_mid), floor)
        elif num_species > SPECIES_TARGET_MAX:
            new_t = max(threshold, t_mid)
        else:
            new_t = max(min(max(threshold, t_lo), t_hi), floor)
        if new_t != threshold:
            config.species_set_config.compatibility_threshold = new_t
            print(f"  [Speciazione] {num_species} specie (obiettivo {SPECIES_TARGET_MIN}-"
                  f"{SPECIES_TARGET_MAX}), threshold {threshold:.2f} -> {new_t:.2f} "
                  f"(fascia {t_lo:.2f}-{t_hi:.2f})")

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
            f"ESC=interrompi | D=dashboard",
            True, (180, 220, 180)
        )
        self.screen.blit(bottom_text, (10, SCREEN_HEIGHT - 26))

    _OVERLAY_PHASES = {
        'validazione': "Selezione dei migliori",
        'conferma': "Conferma del candidato",
        'rivalutazione': "Rivalutazione del record",
    }

    def _draw_validation_overlay(self, done, total):
        """Training visuale: avanzamento della validazione (senza rendering)."""
        self.screen.fill((10, 10, 30))
        text = self._font.render(
            f"Gen {self.generation + 1} | {self._OVERLAY_PHASES.get(self._progress[0], 'Validazione')}: "
            f"episodio {done}/{total} | "
            f"Record val.: {self._format_record()}",
            True, (255, 255, 0)
        )
        self.screen.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)))

    def _format_record(self):
        """Record di validazione per HUD e schermata ('n/d' se non ancora fissato)."""
        return self._fmt(self.best_fitness)

    # ── Dati per la dashboard HTML ──

    def _dashboard_static(self, config):
        """Parametri del training mostrati e spiegati dalla dashboard."""
        stag = config.stagnation_config
        return {
            'fitness_version': FITNESS_VERSION,
            'headless': self._headless,
            'pop_size': config.pop_size,
            'workers': max(1, multiprocessing.cpu_count() - 1) if self._headless else 1,
            'episodes_per_genome': EPISODES_PER_GENOME,
            'validation_seeds': len(VALIDATION_SEEDS),
            'confirmation_seeds': len(CONFIRMATION_SEEDS),
            'validation_top_k': VALIDATION_TOP_K,
            'species_target': [SPECIES_TARGET_MIN, SPECIES_TARGET_MAX],
            'num_inputs': config.genome_config.num_inputs,
            'max_episode_s': MAX_EPISODE_SECONDS,
            'min_episode_s': MIN_EPISODE_SECONDS,
            'early_stop_idle_pct': EARLY_STOP_IDLE_RATIO * 100,
            'early_stop_lives': EARLY_STOP_MAX_LIVES_LOST,
            'early_stop_lives_gen': EARLY_STOP_LIVES_GEN,
            'imitation_start_gen': IMITATION_START_GEN,
            'imitation_max_bonus': IMITATION_MAX_BONUS,
            'max_stagnation': stag.max_stagnation,
            'species_elitism': stag.species_elitism,
            'elitism': config.reproduction_config.elitism,
            'components': [
                {'key': k, 'label': label, 'weight': weight, 'desc': desc}
                for k, label, weight, desc in FITNESS_COMPONENTS
            ],
        }

    def _set_progress(self, phase, done=0, total=0, force=False):
        """Fase e avanzamento correnti, per la schermata e per live.js.

        Fasi: avvio, rivalutazione (record all'avvio), valutazione,
        validazione (selezione), conferma, fermo.
        """
        self._progress = (phase, done, total)
        if self._dash is None:
            return
        self._dash.update_live({
            'phase': phase,
            'done': done,
            'total': total,
            'gen': self.generation,
            'record': self.best_fitness,
            'session_start': int(self._training_start_time * 1000) if self._training_start_time else None,
            'gen_start': int(self._gen_start_time * 1000) if self._gen_start_time else None,
        }, force=force)

    def _open_dashboard(self):
        if self._dash is not None:
            self._dash.open()

    @staticmethod
    def _make_record_info(genome, mean, stats, per_seed, generation):
        """Riepilogo del genoma record per la dashboard (stats = env_stats medi)."""
        info = {
            'key': getattr(genome, 'key', None),
            'value': mean,
            'generation': generation,
            'per_seed': per_seed,
            'seeds': list(CONFIRMATION_SEEDS),
        }
        if stats:
            info['stats'] = _game_stats(stats)
            info['comp'] = {k: stats.get(k, 0) for k, *_ in FITNESS_COMPONENTS}
        return info

    @staticmethod
    def _suite_summary(last, cache):
        """Media, dispersione e statistiche di gioco di una suite giocata
        nell'ultima generazione (`last` = (key, media)), None se non giocata."""
        if not last:
            return None
        key, mean = last
        _, stats, per_seed = cache.get(key, (None, None, None))
        out = {'key': key, 'mean': mean}
        if per_seed and all(f is not None for f in per_seed):
            out['sd'] = _summary(per_seed)['sd']
        if stats:
            out.update(_game_stats(stats))
        return out

    def _validation_summary(self):
        """Migliore della selezione (VALIDATION_SEEDS) della generazione."""
        return self._suite_summary(self._last_validation, self._val_cache)

    def _confirmation_summary(self):
        """Conferma (CONFIRMATION_SEEDS) della generazione, None se non giocata."""
        return self._suite_summary(self._last_confirmation, self._conf_cache)

    def _species_rows(self):
        """Specie della generazione valutata, ordinate per fitness migliore."""
        rows = []
        if self._population is None:
            return rows
        for sid, sp in self._population.species.species.items():
            members = [g for g in sp.members.values() if g.fitness is not None]
            fits = [g.fitness for g in members]
            best = max(members, key=lambda g: g.fitness) if members else None
            rows.append({
                'id': sid,
                'age': self.generation - sp.created,
                'size': len(sp.members),
                'fit_mean': sum(fits) / len(fits) if fits else None,
                'fit_max': best.fitness if best else None,
                'best_key': best.key if best else None,
                'stag': self.generation - sp.last_improved,
            })
        rows.sort(key=lambda r: -math.inf if r['fit_max'] is None else r['fit_max'], reverse=True)
        return rows

    def _generation_report(self, gen_time, max_frames, seeds):
        """Riga dello storico e dettaglio della generazione appena valutata."""
        rows = self._gen_rows
        genome_to_species = self._population.species.genome_to_species if self._population else {}
        ranked = sorted(rows, key=lambda r: r['fitness'], reverse=True)
        top = ranked[:max(1, len(ranked) // 10)]
        comp_keys = [k for k, *_ in FITNESS_COMPONENTS]

        def net_size(genome):
            enabled = [k for k, c in genome.connections.items() if c.enabled]
            # Input distinti letti dalla rete (chiavi negative)
            return len(genome.nodes), len(enabled), len({i for i, _ in enabled if i < 0})

        sizes = [net_size(r['genome']) for r in rows]
        species = self._species_rows()
        best = ranked[0]
        record = {
            'gen': self.generation,
            'ts': time.time(),
            'fv': FITNESS_VERSION,
            'secs': gen_time,
            'pop': len(rows),
            'max_s': max_frames / FPS,
            'fit': _summary([r['fitness'] for r in rows]),
            'imit': _summary([r['stats']['comp_imitation'] for r in rows]),
            'pop_stats': _mean_dicts([_game_stats(r['stats']) for r in rows]),
            'top_stats': _mean_dicts([_game_stats(r['stats']) for r in top]),
            'comp_pop': {k: sum(r['stats'].get(k, 0) for r in rows) / len(rows) for k in comp_keys},
            'gen_best': dict(_game_stats(best['stats']), key=best['genome'].key,
                             fitness=best['fitness'],
                             species=genome_to_species.get(best['genome'].key)),
            'val': self._validation_summary(),
            'conf': self._confirmation_summary(),
            'record': self.best_fitness,
            'record_key': getattr(self.best_genome, 'key', None),
            'new_record': self._new_record_this_gen,
            'record_stats': (self._record_info or {}).get('stats'),
            'species': {
                'n': len(species),
                'threshold': self.config.species_set_config.compatibility_threshold,
                'max_stag': max((r['stag'] for r in species), default=0),
                'sizes': {str(r['id']): r['size'] for r in species},
            },
            'net': {
                'nodes_mean': sum(n for n, _, _ in sizes) / len(sizes),
                'conns_mean': sum(c for _, c, _ in sizes) / len(sizes),
                'nodes_max': max(n for n, _, _ in sizes),
                'conns_max': max(c for _, c, _ in sizes),
                'inputs_mean': sum(i for _, _, i in sizes) / len(sizes),
                'inputs_max': max(i for _, _, i in sizes),
            },
        }

        top_genomes = []
        for rank, r in enumerate(ranked[:10], 1):
            nodes, conns, inputs = net_size(r['genome'])
            top_genomes.append(dict(
                _game_stats(r['stats']), rank=rank, key=r['genome'].key,
                species=genome_to_species.get(r['genome'].key),
                fitness=r['fitness'], imitation=r['stats']['comp_imitation'],
                nodes=nodes, conns=conns, inputs=inputs,
            ))
        snapshot = {
            'gen': self.generation,
            'seeds': list(seeds),
            'top_genomes': top_genomes,
            'species_table': species,
            'record': self._record_info,
        }
        return record, snapshot

    # ── Schermata del training headless ──
    # Riepilogo essenziale nella finestra di gioco: grafici, spiegazioni e
    # tabelle sono nella dashboard HTML (tasto D)

    _C_BG = (13, 16, 25)
    _C_CARD = (24, 29, 44)
    _C_TEXT = (232, 235, 243)
    _C_MUTED = (140, 149, 172)
    _C_GRID = (44, 51, 72)
    _C_ACCENT = (255, 205, 60)
    _C_GOOD = (88, 204, 132)
    _C_BAD = (238, 102, 102)
    _C_INFO = (102, 162, 255)

    _PHASE_LABELS = {
        'avvio': "Avvio del training",
        'rivalutazione': "Rivalutazione del record sulle partite di conferma",
        'valutazione': "Valutazione dei genomi della generazione",
        'validazione': "Selezione dei migliori su partite fisse",
        'conferma': "Conferma del candidato al record",
        'fermo': "Training fermo",
    }

    @staticmethod
    def _fmt(value, decimals=0):
        """Numero con separatore delle migliaia italiano ('n/d' se assente)."""
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return "n/d"
        text = f"{value:,.{decimals}f}"
        return text.replace(",", "_").replace(".", ",").replace("_", ".")

    def _current_history(self):
        """Generazioni dello storico con la versione di fitness corrente."""
        if self._dash is None:
            return []
        return [r for r in self._dash.history if r.get('fv') == FITNESS_VERSION]

    def _draw_stats_screen(self):
        """Schermata del training headless: fase, indicatori chiave, andamento."""
        s = self.screen
        f = self._fonts
        s.fill(self._C_BG)
        x0, x1 = 40, SCREEN_WIDTH - 40

        # Intestazione
        s.blit(f['title'].render("Training dell'AI", True, self._C_TEXT), (x0, 20))
        sub = f"Generazione {self.generation}"
        if self._training_start_time:
            elapsed = time.time() - self._training_start_time
            sub += f"   ·   sessione {self._format_time(elapsed)}"
            gens = self.generation - self._start_generation
            if gens > 0 and elapsed > 0:
                sub += f"   ·   {self._fmt(gens / (elapsed / 60), 1)} generazioni/min"
        s.blit(f['body'].render(sub, True, self._C_MUTED), (x0, 64))

        # Fase e avanzamento
        phase, done, total = self._progress
        label = self._PHASE_LABELS.get(phase, phase)
        s.blit(f['label'].render(label, True, self._C_TEXT), (x0, 104))
        if total:
            count = f['label'].render(f"{done} / {total}", True, self._C_MUTED)
            s.blit(count, count.get_rect(topright=(x1, 104)))
        bar = pygame.Rect(x0, 130, x1 - x0, 12)
        pygame.draw.rect(s, self._C_CARD, bar, border_radius=6)
        if total:
            fill = bar.copy()
            fill.width = max(12, int(bar.width * done / total))
            color = self._C_INFO if phase in ('validazione', 'conferma', 'rivalutazione') else self._C_GOOD
            pygame.draw.rect(s, color, fill, border_radius=6)

        hist = self._current_history()
        self._draw_kpi_tiles(x0, 166, x1, hist)
        self._draw_progress_chart(pygame.Rect(x0, 330, x1 - x0, 300), hist)

        # Piede: comandi
        y = SCREEN_HEIGHT - 84
        pygame.draw.line(s, self._C_GRID, (x0, y), (x1, y), 1)
        s.blit(f['body'].render("D  apre la dashboard dettagliata nel browser (grafici, spiegazioni, tabelle)",
                                True, self._C_ACCENT), (x0, y + 14))
        esc = f['body'].render("ESC  ferma il training e salva il checkpoint", True, self._C_TEXT)
        s.blit(esc, (x0, y + 42))
        path = os.path.join(DASHBOARD_DIR, "index.html")
        max_w = x1 - (x0 + esc.get_width() + 40)
        while len(path) > 10 and f['small'].size(path)[0] > max_w:
            path = "…" + path[2:]
        surf = f['small'].render(path, True, self._C_MUTED)
        s.blit(surf, surf.get_rect(topright=(x1, y + 46)))

    def _draw_kpi_tiles(self, x0, y, x1, hist):
        """Quattro indicatori: record, record in partita, popolazione, specie."""
        f = self._fonts
        gap = 20
        w = (x1 - x0 - 3 * gap) // 4
        h = 140

        # 1. Record di validazione e variazione nella sessione
        if self.best_fitness == -math.inf:
            delta = ("in attesa della prima validazione", self._C_MUTED)
        elif self._session_start_record == -math.inf:
            delta = ("primo record di questa sessione", self._C_GOOD)
        elif self.best_fitness > self._session_start_record:
            delta = (f"+{self._fmt(self.best_fitness - self._session_start_record)} in questa sessione",
                     self._C_GOOD)
        else:
            delta = ("nessun nuovo record in sessione", self._C_MUTED)
        rec_stats = (self._record_info or {}).get('stats')
        tiles = [("RECORD DI VALIDAZIONE", self._format_record(), self._C_ACCENT,
                  f"fitness media su {len(CONFIRMATION_SEEDS)} partite fisse", delta)]

        # 2. Come gioca il record
        if rec_stats:
            tiles.append(("IL RECORD IN PARTITA", f"{self._fmt(rec_stats['score'])} punti", self._C_TEXT,
                          f"sopravvive {self._fmt(rec_stats['surv'])} s, livello {self._fmt(rec_stats['level'], 1)}",
                          (f"{self._fmt(rec_stats['kills'] + rec_stats['missile_kills'] + rec_stats['bomb_kills'], 1)}"
                           f" uccisioni, {self._fmt(rec_stats['lives_lost'], 1)} vite perse", self._C_MUTED)))
        else:
            tiles.append(("IL RECORD IN PARTITA", "n/d", self._C_TEXT,
                          f"media delle {len(CONFIRMATION_SEEDS)} partite", ("", self._C_MUTED)))

        # 3. Fitness media della popolazione (media mobile) e tendenza
        means = [r['fit']['mean'] for r in hist if r.get('fit')]
        if means:
            rolling = sum(means[-10:]) / len(means[-10:])
            tr = _trend(means)
            if tr is None:
                trend = (f"tendenza tra {8 - len(means)} generazioni", self._C_MUTED)
            elif tr[0] >= 2:
                trend = (f"in miglioramento ({tr[1]:+.0f}%)", self._C_GOOD)
            elif tr[0] <= -2:
                trend = (f"in calo ({tr[1]:+.0f}%)", self._C_BAD)
            else:
                trend = ("stabile, entro il rumore", self._C_MUTED)
            tiles.append(("FITNESS MEDIA POPOLAZIONE", self._fmt(rolling), self._C_TEXT,
                          "media mobile di 10 generazioni", trend))
        else:
            tiles.append(("FITNESS MEDIA POPOLAZIONE", "n/d", self._C_TEXT,
                          "media mobile di 10 generazioni", ("", self._C_MUTED)))

        # 4. Specie
        if self._population is not None:
            n_species = len(self._population.species.species)
            stag = max((self.generation - sp.last_improved
                        for sp in self._population.species.species.values()), default=0)
            max_stag = self.config.stagnation_config.max_stagnation
            tiles.append(("SPECIE", str(n_species), self._C_TEXT,
                          f"soglia di compatibilità {self._fmt(self.config.species_set_config.compatibility_threshold, 2)}",
                          (f"stallo massimo {stag} su {max_stag} generazioni",
                           self._C_BAD if stag >= max_stag - 3 else self._C_MUTED)))

        for i, (title, value, vcolor, line1, (line2, color2)) in enumerate(tiles):
            rect = pygame.Rect(x0 + i * (w + gap), y, w, h)
            pygame.draw.rect(self.screen, self._C_CARD, rect, border_radius=10)
            self.screen.blit(f['label'].render(title, True, self._C_MUTED), (rect.x + 16, rect.y + 14))
            self.screen.blit(f['kpi'].render(value, True, vcolor), (rect.x + 16, rect.y + 36))
            self.screen.blit(f['small'].render(line1, True, self._C_MUTED), (rect.x + 16, rect.y + 88))
            self.screen.blit(f['small'].render(line2, True, color2), (rect.x + 16, rect.y + 110))

    def _draw_progress_chart(self, rect, hist):
        """Andamento: record, migliore della selezione per generazione, media mobile."""
        s = self.screen
        f = self._fonts
        pygame.draw.rect(s, self._C_CARD, rect, border_radius=10)
        s.blit(f['label'].render("ANDAMENTO DELLA FITNESS (ultime 200 generazioni)", True, self._C_MUTED),
               (rect.x + 16, rect.y + 12))
        hist = hist[-200:]
        if len(hist) < 2:
            msg = f['body'].render("Il grafico compare dopo due generazioni completate.", True, self._C_MUTED)
            s.blit(msg, msg.get_rect(center=rect.center))
            return

        legend = [("record", self._C_GOOD), ("migliore della selezione", self._C_INFO),
                  ("media della popolazione (mobile, 10 gen)", self._C_ACCENT)]
        lx = rect.right - 16
        for text, color in reversed(legend):
            surf = f['small'].render(text, True, self._C_TEXT)
            lx -= surf.get_width()
            s.blit(surf, (lx, rect.y + 14))
            lx -= 18
            pygame.draw.circle(s, color, (lx + 7, rect.y + 23), 5)
            lx -= 14

        gens = [r['gen'] for r in hist]
        means = [r['fit']['mean'] for r in hist]
        rolling = [sum(means[max(0, i - 9):i + 1]) / len(means[max(0, i - 9):i + 1])
                   for i in range(len(means))]
        records = [r.get('record') for r in hist]
        vals = [(r.get('val') or {}).get('mean') for r in hist]
        ys = [v for v in rolling + records + vals if v is not None]
        lo, hi = min(ys), max(ys)
        if hi - lo < 1:
            lo, hi = lo - 1, hi + 1
        # Passo "tondo" (1, 2, 2.5, 5 x 10^n) per 4 intervalli
        raw = (hi - lo) / 4
        mag = 10 ** math.floor(math.log10(raw))
        step = next(m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag + 1e-9)
        lo, hi = math.floor(lo / step) * step, math.ceil(hi / step) * step
        ticks = int(round((hi - lo) / step))
        plot = pygame.Rect(rect.x + 70, rect.y + 46, rect.width - 90, rect.height - 80)
        g0, g1 = gens[0], gens[-1]

        def px(g):
            return plot.x + (g - g0) / (g1 - g0) * plot.width

        def py(v):
            return plot.bottom - (v - lo) / (hi - lo) * plot.height

        for i in range(ticks + 1):
            v = lo + step * i
            y = py(v)
            pygame.draw.line(s, self._C_GRID, (plot.x, y), (plot.right, y), 1)
            lab = f['small'].render(self._fmt(v, 0 if step >= 1 else 1), True, self._C_MUTED)
            s.blit(lab, lab.get_rect(midright=(plot.x - 8, y)))
        for g in (g0, (g0 + g1) // 2, g1):
            lab = f['small'].render(f"gen {g}", True, self._C_MUTED)
            s.blit(lab, lab.get_rect(midtop=(px(g), plot.bottom + 6)))

        pts = [(px(g), py(v)) for g, v in zip(gens, rolling)]
        pygame.draw.lines(s, self._C_ACCENT, False, pts, 3)
        for g, v in zip(gens, vals):
            if v is not None:
                pygame.draw.circle(s, self._C_INFO, (px(g), py(v)), 3)
        step = []
        for g, v in zip(gens, records):
            if v is None:
                continue
            if step:
                step.append((px(g), step[-1][1]))
            step.append((px(g), py(v)))
        if len(step) > 1:
            pygame.draw.lines(s, self._C_GOOD, False, step, 3)

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
                obs_version = session.get('obs_version', 1)
                if obs_version != OBS_VERSION:
                    raise ValueError(f"osservazione v{obs_version}, attuale v{OBS_VERSION}")
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
        num_inputs = self.config.genome_config.num_inputs
        for _generation, path in reversed(_list_checkpoints()):
            try:
                with gzip.open(path) as f:
                    data = pickle.load(f)
            except Exception as e:
                print(f"Checkpoint illeggibile, ignorato: {path} ({e!r})")
                continue
            # Popolazione allenata su un altro layout dell'osservazione
            obs_version = getattr(data[1], 'obs_version', 1)
            if obs_version != OBS_VERSION or data[1].genome_config.num_inputs != num_inputs:
                print(f"Checkpoint per l'osservazione v{obs_version}, attuale v{OBS_VERSION}: "
                      f"ignorato ({path})")
                continue
            return path, data[1]
        return None, None

class _TrainingInterrupted(Exception):
    """Eccezione interna per interrompere il training con ESC."""
    pass
