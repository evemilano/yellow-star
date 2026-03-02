"""Orchestrazione training NEAT per Yellow Star.

Gestisce:
- Evoluzione della popolazione NEAT
- Valutazione fitness dei genomi (con rendering live)
- Bonus imitation dalle sessioni umane
- Persistenza (salva/carica genomi e checkpoint)
- Modalita' "watch" per vedere il miglior genoma giocare
"""

import glob
import os
import pickle
import time
import multiprocessing
import neat
import pygame
from settings import SCREEN_WIDTH, SCREEN_HEIGHT, FPS, PLAYER_LIVES
from src.game_env import GameEnv

AI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ai")
CONFIG_PATH = os.path.join(AI_DIR, "neat_config.txt")
BEST_GENOME_PATH = os.path.join(AI_DIR, "best_genome.pkl")
CHECKPOINT_PREFIX = os.path.join(AI_DIR, "neat-checkpoint-")
MAX_CHECKPOINTS = 2  # mantieni solo gli ultimi N checkpoint
SESSIONS_DIR = os.path.join(AI_DIR, "human_sessions")

MAX_FRAMES_PER_EPISODE = 5400  # 3 minuti a 30 FPS
MIN_FRAMES_PER_EPISODE = 900   # 30 secondi minimo

# Velocita' di rendering durante il training visuale
TRAIN_RENDER_SKIP = 4  # mostra 1 frame su 4

# Terminazione anticipata genomi scarsi (check periodico)
EARLY_STOP_INTERVAL = 150      # controlla ogni 5 secondi
EARLY_STOP_MAX_LIVES_LOST = 2  # se ha gia' perso 2 vite, interrompi
EARLY_STOP_IDLE_RATIO = 0.95   # se fermo per il 95% del tempo, interrompi
EARLY_STOP_MIN_FRAME = 150     # non terminare prima di 5 secondi

# Generazioni prima di attivare l'imitation bonus
IMITATION_START_GEN = 10


def _compute_fitness(env):
    """Calcola fitness e statistiche da un GameEnv dopo la simulazione.

    Funzione top-level (picklable) usata sia dal worker che da _eval_single.
    Ritorna (fitness, env_stats).
    """
    idle_ratio = env.idle_frames / max(1, env.frame_count)
    accuracy_bonus = 0.0
    if env.shots_fired > 0:
        accuracy = env.shots_hit / env.shots_fired
        accuracy_bonus = accuracy * env.enemies_killed * 10

    fitness = (
        env.score * 2.0
        + env.enemies_killed * 30
        + env.dodges * 5
        + accuracy_bonus
        + env.frame_count * 0.05
        - env.lives_lost * 200
        + env.level_manager.level * 100
        + env.powerups_collected * 50
        + env.missiles_collected * 30
        + env.total_upgrade_levels * 20
        - env.shield_hits * 30
        - env.shield_breaks * 80
        - idle_ratio * env.frame_count * 0.5
        + env.near_misses * 3
        + env.powerup_approach_frames * 0.1
    )
    # NON clippare a 0: i genomi pessimi devono avere fitness negativa

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
        'dodges': env.dodges,
        'accuracy': env.shots_hit / max(1, env.shots_fired),
        'near_misses': env.near_misses,
        'comp_score': env.score * 2.0,
        'comp_kills': env.enemies_killed * 30,
        'comp_dodges': env.dodges * 5,
        'comp_accuracy': accuracy_bonus,
        'comp_frames': env.frame_count * 0.05,
        'comp_lives': -env.lives_lost * 200,
        'comp_level': env.level_manager.level * 100,
        'comp_powerups': env.powerups_collected * 50,
        'comp_missiles': env.missiles_collected * 30,
        'comp_upgrades': env.total_upgrade_levels * 20,
        'comp_shield': -(env.shield_hits * 30 + env.shield_breaks * 80),
        'comp_idle': -(idle_ratio * env.frame_count * 0.5),
        'comp_near_misses': env.near_misses * 3,
        'comp_pu_approach': env.powerup_approach_frames * 0.1,
    }

    return fitness, env_stats


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
        'shoot': output[2] > 0.0,
        'missile': output[3] > 0.5,
        'bomb': output[4] > 0.5,
    }


def _init_worker():
    """Inizializza pygame in ogni processo worker del Pool."""
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    pygame.display.set_mode((1, 1))


def _eval_genome_worker(args):
    """Valuta un singolo genoma in un processo separato (per multiprocessing).

    Funzione top-level per essere picklable.
    Restituisce (idx, fitness, env_stats) per imap_unordered.
    """
    idx, _genome_id, genome, config, max_frames, generation, seed = args

    try:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        env = GameEnv(render=False, screen=None)
        obs = env.reset(seed=seed)

        for frame in range(max_frames):
            output = net.activate(obs)
            actions = _decode_actions_static(output)
            obs, reward, done = env.step(actions)

            # Terminazione anticipata
            if frame >= EARLY_STOP_MIN_FRAME and frame % EARLY_STOP_INTERVAL == 0:
                max_lives = EARLY_STOP_MAX_LIVES_LOST if generation >= 10 else PLAYER_LIVES
                if env.lives_lost >= max_lives:
                    break
                if env.idle_frames / max(1, frame) >= EARLY_STOP_IDLE_RATIO:
                    break

            if done:
                break

        fitness, env_stats = _compute_fitness(env)
        env.close()
        return idx, fitness, env_stats
    except Exception:
        # Ritorna fitness molto negativa per non bloccare il training
        return idx, -10000.0, {
            'score': 0, 'frame_count': 0, 'level': 1,
            'lives_lost': 0, 'powerups': 0, 'missiles': 0,
            'upgrades': 0, 'idle_frames': 0, 'enemies_killed': 0,
            'dodges': 0, 'accuracy': 0, 'near_misses': 0,
            'comp_score': 0, 'comp_kills': 0, 'comp_dodges': 0,
            'comp_accuracy': 0, 'comp_frames': 0, 'comp_lives': 0,
            'comp_level': 0, 'comp_powerups': 0, 'comp_missiles': 0,
            'comp_upgrades': 0, 'comp_shield': 0, 'comp_idle': 0,
            'comp_near_misses': 0, 'comp_pu_approach': 0,
        }


class _RollingCheckpointer(neat.Checkpointer):
    """Checkpointer che mantiene solo gli ultimi MAX_CHECKPOINTS file."""

    def save_checkpoint(self, config, population, species_set, generation):
        super().save_checkpoint(config, population, species_set, generation)
        self._cleanup_old_checkpoints()

    @staticmethod
    def _cleanup_old_checkpoints():
        pattern = CHECKPOINT_PREFIX + "*"
        files = glob.glob(pattern)
        if len(files) <= MAX_CHECKPOINTS:
            return
        files.sort(key=lambda f: int(f.split("-")[-1]))
        for old in files[:-MAX_CHECKPOINTS]:
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
        self.best_fitness = -1
        self.generation = 0
        self._human_sessions = None  # cache lazy
        self._headless = False
        self._font = None
        self._font_small = None
        self._font_big = None
        self._font_section = None
        # Riferimenti NEAT (set in train())
        self._population = None
        self._stats_reporter = None
        # Statistiche per schermata headless
        self._last_gen_stats = None
        self._current_gen_fitnesses = []
        self._training_start_time = None
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
        # Scomposizione fitness miglior genoma
        self._best_genome_components = None
        self._current_gen_fitness_components = []
        self._new_record_this_gen = False
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

        # In headless: popolazione piu' grande per piu' diversita'
        if headless:
            self.config.pop_size = 200

        # Cerca checkpoint per riprendere
        checkpoint = self._find_latest_checkpoint()
        if checkpoint:
            print(f"Ripresa dal checkpoint: {checkpoint}")
            p = neat.Checkpointer.restore_checkpoint(checkpoint)
            # Applica pop_size anche al checkpoint ripristinato
            if headless:
                p.config.pop_size = 200
            # Sincronizza generazione interna con il checkpoint
            self.generation = p.generation
            print(f"  Generazione ripristinata: {self.generation}")
        else:
            print(f"Inizio nuovo training (pop_size={self.config.pop_size})")
            p = neat.Population(self.config)

        # Carica miglior genoma precedente se esiste
        if os.path.exists(BEST_GENOME_PATH):
            with open(BEST_GENOME_PATH, 'rb') as f:
                self.best_genome = pickle.load(f)
                self.best_fitness = self.best_genome.fitness or 0

        # Reporter
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(_RollingCheckpointer(
            generation_interval=10,
            filename_prefix=CHECKPOINT_PREFIX,
        ))

        self._population = p
        self._stats_reporter = stats

        # Pool persistente per headless (evita di ricreare i processi ogni gen)
        if headless:
            num_workers = max(1, multiprocessing.cpu_count() - 1)
            self._pool = multiprocessing.Pool(
                processes=num_workers, initializer=_init_worker
            )

        try:
            p.run(self._eval_genomes, generations)
        except _TrainingInterrupted:
            print("\nTraining interrotto dall'utente.")
        finally:
            if self._pool:
                self._pool.terminate()
                self._pool.join()
                self._pool = None

        # Il miglior genoma e' gia' salvato incrementalmente in _eval_genomes
        if self.best_genome:
            print(f"\nMiglior fitness raggiunta: {self.best_fitness:.1f}")

    def _eval_genomes(self, genomes, config):
        """Valuta tutti i genomi della popolazione.

        In modalita' headless usa multiprocessing per parallelizzare.
        """
        # Pre-carica sessioni umane (una volta)
        if self._human_sessions is None:
            self._human_sessions = self._load_human_sessions()

        self._gen_start_time = time.time()

        best_in_gen = None
        best_in_gen_idx = 0
        best_fitness_in_gen = -float('inf')
        total = len(genomes)
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

        max_frames = self._get_max_frames()

        if self._headless:
            # ── Valutazione PARALLELA (headless, pool persistente) ──
            worker_args = [
                (idx, genome_id, genome, config, max_frames, self.generation,
                 genome_id + self.generation * 10000)
                for idx, (genome_id, genome) in enumerate(genomes)
            ]

            # imap_unordered: i worker non restano mai idle in attesa
            # dei genomi piu' lenti — appena uno finisce, parte il successivo
            completed = 0
            for idx, fitness, env_stats in self._pool.imap_unordered(
                _eval_genome_worker, worker_args, chunksize=1
            ):
                genome_id, genome = genomes[idx]
                genome.fitness = fitness
                self._collect_stats(genome, env_stats)

                if genome.fitness > best_fitness_in_gen:
                    best_fitness_in_gen = genome.fitness
                    best_in_gen = genome
                    best_in_gen_idx = idx

                completed += 1
                # Aggiorna dashboard durante la valutazione
                if self.screen and completed % 5 == 0:
                    self._draw_stats_screen(completed, total, config)
                    pygame.display.flip()
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise _TrainingInterrupted()
                        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                            raise _TrainingInterrupted()

            # Aggiorna schermata finale
            if self.screen:
                self._draw_stats_screen(total, total, config)
                pygame.display.flip()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise _TrainingInterrupted()
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        raise _TrainingInterrupted()
        else:
            # ── Valutazione SERIALE (con rendering) ──
            for idx, (genome_id, genome) in enumerate(genomes):
                fitness, env_stats = self._eval_single(
                    genome, config, idx + 1, total, render
                )
                genome.fitness = fitness
                self._collect_stats(genome, env_stats)

                if genome.fitness > best_fitness_in_gen:
                    best_fitness_in_gen = genome.fitness
                    best_in_gen = genome
                    best_in_gen_idx = idx

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

        # Salva il miglior genoma se migliore del record
        new_record = False
        if best_in_gen and best_fitness_in_gen > self.best_fitness:
            self.best_fitness = best_fitness_in_gen
            self.best_genome = best_in_gen
            self._best_genome_components = self._current_gen_fitness_components[best_in_gen_idx]
            new_record = True
            os.makedirs(AI_DIR, exist_ok=True)
            with open(BEST_GENOME_PATH, 'wb') as f:
                pickle.dump(best_in_gen, f)

        self._new_record_this_gen = new_record
        self.generation += 1

        # Check anti-stagnazione automatico
        self._check_and_fix_stagnation(config)

        # In headless: aggiorna schermata fine generazione
        if self._headless and self.screen:
            self._draw_stats_screen(total, total, config)
            pygame.display.flip()

    def _get_max_frames(self):
        """Frame massimi adattivi: piu' generazioni = piu' tempo concesso."""
        # Cresce da MIN a MAX in 50 generazioni
        t = min(1.0, self.generation / 50.0)
        return int(MIN_FRAMES_PER_EPISODE + t * (MAX_FRAMES_PER_EPISODE - MIN_FRAMES_PER_EPISODE))

    def _collect_stats(self, genome, env_stats):
        """Raccoglie le statistiche di un genoma valutato.

        Applica anche l'imitation bonus se disponibile.
        """
        if self._human_sessions and self.generation >= IMITATION_START_GEN:
            genome.fitness += self._imitation_bonus(genome, self.config)
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

    def _eval_single(self, genome, config, genome_num=0, total=0, render=False):
        """Valuta un singolo genoma giocando una partita.

        Se render=True, mostra la partita sullo schermo in tempo reale.
        """
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        env = GameEnv(render=render, screen=self.screen)
        seed = (genome.key if hasattr(genome, 'key') else id(genome)) + self.generation * 10000
        obs = env.reset(seed=seed)
        max_frames = self._get_max_frames()

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

            output = net.activate(obs)
            actions = self._decode_actions(output)
            obs, reward, done = env.step(actions)

            # Terminazione anticipata: genoma scarso (check periodico)
            if (frame >= EARLY_STOP_MIN_FRAME
                    and frame % EARLY_STOP_INTERVAL == 0):
                # Nelle prime 10 generazioni, non terminare per vite perse
                # (i genomi iniziali sono tutti terribili, servono dati)
                max_lives = EARLY_STOP_MAX_LIVES_LOST if self.generation >= 10 else PLAYER_LIVES
                if env.lives_lost >= max_lives:
                    break
                # Troppo inattivo (sta fermo e basta)
                if env.idle_frames / max(1, frame) >= EARLY_STOP_IDLE_RATIO:
                    break

            # Rendering: mostra 1 frame su TRAIN_RENDER_SKIP
            if render and frame % TRAIN_RENDER_SKIP == 0:
                self._draw_training_overlay(
                    env, genome_num, total, frame
                )
                pygame.display.flip()

            if done:
                break

        fitness, env_stats = _compute_fitness(env)
        env.close()
        return fitness, env_stats

    def _check_and_fix_stagnation(self, config):
        """Rileva stagnazione e interviene automaticamente."""
        if not self._population:
            return

        species_set = self._population.species
        num_species = len(species_set.species)

        # Se 1 sola specie: abbassa threshold per forzare speciazione
        if num_species <= 1:
            threshold = config.species_set_config.compatibility_threshold
            if threshold > 0.5:
                new_t = max(0.5, threshold * 0.9)
                config.species_set_config.compatibility_threshold = new_t
                print(f"  [Anti-stagnazione] Threshold abbassato: {threshold:.2f} -> {new_t:.2f}")

        # Se fitness media non migliora da 30 gen: inietta diversita'
        if len(self._fitness_history) >= 30:
            recent_avgs = [h[0] for h in self._fitness_history[-30:]]
            improvement = max(recent_avgs) - min(recent_avgs)
            pct = improvement / max(1, abs(recent_avgs[0])) * 100
            if pct < 5:  # meno del 5% di miglioramento in 30 gen
                self._inject_fresh_genomes(config)

    def _inject_fresh_genomes(self, config):
        """Sostituisce il 10% peggiore con genomi freschi random."""
        pop = self._population.population
        num_replace = max(5, len(pop) // 10)

        sorted_genomes = sorted(
            pop.items(), key=lambda x: x[1].fitness or -float('inf')
        )
        for gid, _ in sorted_genomes[:num_replace]:
            del pop[gid]

        new_id = max(pop.keys()) + 1
        for i in range(num_replace):
            g = config.genome_type(new_id + i)
            g.configure_new(config.genome_config)
            pop[new_id + i] = g

        print(f"  [Anti-stagnazione] Iniettati {num_replace} genomi freschi")

    def _draw_training_overlay(self, env, genome_num, total, frame):
        """Disegna l'overlay con info training sopra il gioco."""
        overlay_bg = pygame.Surface((SCREEN_WIDTH, 50), pygame.SRCALPHA)
        overlay_bg.fill((0, 0, 0, 160))
        self.screen.blit(overlay_bg, (0, 0))

        # Riga superiore: info generazione e genoma
        n_sessions = len(self._human_sessions) if self._human_sessions else 0
        imitation = f" | Sessioni umane: {n_sessions}" if n_sessions > 0 else ""

        top_text = self._font.render(
            f"Gen {self.generation + 1} | "
            f"Genoma {genome_num}/{total} | "
            f"Record: {self.best_fitness:.0f}"
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

        return y + 6

    # ── Sezione B: Fitness ──

    def _draw_section_fitness(self, x, y):
        y = self._draw_section_header("FITNESS", x, y)

        # Record
        record_color = (100, 255, 100)
        record_text = f"{self.best_fitness:.0f}"
        if hasattr(self, '_new_record_this_gen') and self._new_record_this_gen:
            record_text += "  *NUOVO*"
        self._draw_kv(x, y, "Record:", record_text, record_color)
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

        items = [
            ("Score",      comp['comp_score'],                      (100, 255, 100)),
            ("Kills",      comp.get('comp_kills', 0),               (120, 255, 120)),
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

        # Tempo totale + gen/min su una riga
        if self._training_start_time:
            elapsed = time.time() - self._training_start_time
            eff_text = self._format_time(elapsed)
            if self.generation > 0:
                eff_text += f"  |  {self.generation / (elapsed / 60.0):.1f} gen/min"
            self._draw_kv(x, y, "Tempo:", eff_text, (200, 200, 100))
            y += 20

        # Tempo/gen medio + ultimo su una riga
        if self._gen_times:
            avg_t = sum(self._gen_times) / len(self._gen_times)
            last_t = self._gen_times[-1]
            t_color = (100, 255, 100) if last_t <= avg_t else (255, 200, 100)
            self._draw_kv(x, y, "Tempo/gen:", f"avg {avg_t:.1f}s  last {last_t:.1f}s", t_color)
            y += 20

        # Genomi totali
        pop = config.pop_size if config else 100
        total_genomes = self.generation * pop + genome_num
        self._draw_kv(x, y, "Genomi valutati:", str(total_genomes), (180, 180, 200))
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
        """Bonus fitness per somiglianza alle azioni umane registrate."""
        if not self._human_sessions:
            return 0.0

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        total_match = 0
        total_weight = 0

        # Usa le ultime 5 sessioni, pesate per score
        for session in self._human_sessions[-5:]:
            score_weight = max(1.0, session['score'] / 500.0)
            frames = session['frames']

            # Campiona max 200 frame per sessione (uniformemente)
            step = max(1, len(frames) // 200)
            for j in range(0, len(frames), step):
                obs, human_action = frames[j]
                ai_output = net.activate(obs)
                ai_action = self._decode_actions(ai_output)

                match = self._action_similarity(ai_action, human_action)
                total_match += match * score_weight
                total_weight += score_weight

        if total_weight == 0:
            return 0.0

        # Max bonus ~300 punti
        similarity = total_match / total_weight
        return similarity * 300

    @staticmethod
    def _action_similarity(ai_action, human_action):
        """Calcola somiglianza tra azione AI e azione umana (0-1)."""
        score = 0.0
        total = 5.0  # 5 componenti

        # Movimento verticale (match esatto = 1, off-by-one = 0.5)
        v_diff = abs(ai_action['vertical'] - human_action['vertical'])
        if v_diff == 0:
            score += 1.0
        elif v_diff == 1:
            score += 0.5

        # Movimento orizzontale
        h_diff = abs(ai_action['horizontal'] - human_action['horizontal'])
        if h_diff == 0:
            score += 1.0
        elif h_diff == 1:
            score += 0.5

        # Sparo (match esatto)
        if ai_action['shoot'] == human_action['shoot']:
            score += 1.0

        # Missile (match esatto, retrocompat sessioni senza campo)
        if ai_action['missile'] == human_action.get('missile', False):
            score += 1.0

        # Bomba (match esatto, retrocompat sessioni senza campo)
        if ai_action['bomb'] == human_action.get('bomb', False):
            score += 1.0

        return score / total

    @staticmethod
    def _decode_actions(output):
        """Converte output della rete in azioni di gioco."""
        return _decode_actions_static(output)

    def _load_human_sessions(self):
        """Carica le sessioni umane salvate su disco."""
        if not os.path.exists(SESSIONS_DIR):
            return []

        sessions = []
        for filename in sorted(os.listdir(SESSIONS_DIR)):
            if not filename.endswith('.pkl'):
                continue
            filepath = os.path.join(SESSIONS_DIR, filename)
            try:
                with open(filepath, 'rb') as f:
                    session = pickle.load(f)
                sessions.append(session)
            except (pickle.UnpicklingError, EOFError, KeyError):
                continue

        return sessions

    def _find_latest_checkpoint(self):
        """Trova il checkpoint NEAT piu' recente."""
        if not os.path.exists(AI_DIR):
            return None

        checkpoints = [
            f for f in os.listdir(AI_DIR)
            if f.startswith("neat-checkpoint-")
        ]
        if not checkpoints:
            return None

        checkpoints.sort(
            key=lambda f: int(f.split("-")[-1])
        )
        return os.path.join(AI_DIR, checkpoints[-1])

class _TrainingInterrupted(Exception):
    """Eccezione interna per interrompere il training con ESC."""
    pass
