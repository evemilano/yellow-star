"""Ambiente di gioco steppabile per il training NEAT.

Replica la logica di Game._update() ma controllabile dall'esterno:
- step(actions) avanza di 1 frame
- get_observation() estrae lo stato normalizzato per la rete neurale
- Puo' girare headless (senza rendering) per velocita'
"""

import heapq
import random
import pygame
from settings import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FPS,
    PLAYER_LIVES,
    POWERUP_DROP_CHANCE,
    BOMB_FREEZE_DURATION, BOMB_FIELD_DURATION,
    AI_MISSILE_INTERVAL,
    LEVEL_MAX,
    ENEMY_BULLET_SPEED,
)
from src.player import Player
from src.enemy import Enemy
from src.asteroid import Asteroid
from src.missile import MissilePowerUp, Missile
from src.explosion import Explosion
from src.enemy_bullet import EnemyBullet
from src.level import LevelManager
from src.powerup import PowerUp

MS_PER_FRAME = 1000.0 / FPS  # ~16.7 ms a 60 FPS


def _nearest_n(sprites, px, py, n):
    """Trova gli n sprite piu' vicini a (px, py)."""
    items = sprites.sprites()
    if len(items) <= n:
        items.sort(key=lambda s: (s.rect.centerx - px) ** 2 + (s.rect.centery - py) ** 2)
        return items
    return heapq.nsmallest(
        n, items,
        key=lambda s: (s.rect.centerx - px) ** 2 + (s.rect.centery - py) ** 2
    )


def groupcollide_fast(groupa, groupb, dokilla, dokillb):
    """Equivalente di pygame.sprite.groupcollide (senza callback `collided`).

    Stesso risultato e stesso ordine: sprite di A nell'ordine del gruppo,
    bersagli nell'ordine di B, kill progressivi (un bersaglio ucciso da uno
    sprite di A non e' piu' visibile ai successivi), kill di A alla fine.
    Invece di una spritecollide Python per ogni sprite di A usa
    Rect.collidelistall sui rect di B letti una sola volta.
    Usata sia da GameEnv che da Game: tenerle allineate.
    """
    targets = groupb.sprites()
    if not targets:
        return {}
    rects = [s.rect for s in targets]
    alive = [True] * len(targets)
    hits = {}
    for a in groupa.sprites():
        hit = []
        for i in a.rect.collidelistall(rects):
            if alive[i]:
                hit.append(targets[i])
                if dokillb:
                    alive[i] = False
        if hit:
            if dokillb:
                for s in hit:
                    s.kill()
            hits[a] = hit
    if dokilla:
        for a in hits:
            a.kill()
    return hits


def build_observation(player, enemies, asteroids, enemy_bullets, powerups,
                      level_manager, now_ms):
    """Estrae 52 valori normalizzati dallo stato di gioco.

    Funzione standalone usata sia da GameEnv che da HumanRecorder
    per garantire coerenza nel vettore di osservazione.

    Args:
        player: Player con rect, lives, shield_hp, upgrades, missiles, bombs, last_shot
        enemies: sprite Group di nemici
        asteroids: sprite Group di asteroidi
        enemy_bullets: sprite Group di proiettili nemici
        powerups: sprite Group di power-up
        level_manager: LevelManager con .level
        now_ms: tempo corrente in millisecondi (virtuale o reale)

    Layout:
      [0..7]   Player state base (8)
      [8..13]  Player state esteso: cooldown, densita', upgrade (6)
      [14..25] 3 nemici: dx, dy, speed, is_shooter (12)
      [26..37] 3 asteroidi: dx, dy, speed, size (12)
      [38..49] 3 proiettili nemici: dx, dy, vx, vy (12)
      [50..51] 1 power-up piu' vicino: dx, dy (2)
    Totale: 52
    """
    px = player.rect.centerx
    py = player.rect.centery

    # Player state base (8 valori)
    shield_max = max(1, player.upgrades.shield_max)
    obs = [
        px / SCREEN_WIDTH,
        py / SCREEN_HEIGHT,
        player.lives / PLAYER_LIVES,
        player.shield_hp / shield_max if shield_max > 0 else 0.0,
        1.0 if player.invincible else 0.0,
        min(player.missiles, 3) / 3.0,
        min(player.bombs, 3) / 3.0,
        level_manager.level / LEVEL_MAX,
    ]

    # Player state esteso (6 valori)
    shoot_delay = player.upgrades.shoot_delay
    time_since_shot = now_ms - player.last_shot
    shoot_cooldown = max(0.0, 1.0 - time_since_shot / shoot_delay)
    obs.append(shoot_cooldown)
    obs.append(min(len(enemies), 10) / 10.0)
    obs.append(min(len(enemy_bullets), 10) / 10.0)
    obs.append(player.upgrades.levels["weapon"] / 10.0)
    obs.append(player.upgrades.levels["engine"] / 10.0)
    obs.append(player.upgrades.levels["shield"] / 10.0)

    # 3 nemici piu' vicini (4 valori ciascuno: dx, dy, speed, is_shooter)
    nearest_enemies = _nearest_n(enemies, px, py, 3)
    for e in nearest_enemies:
        obs.append((e.rect.centerx - px) / SCREEN_WIDTH)
        obs.append((e.rect.centery - py) / SCREEN_HEIGHT)
        obs.append(e.speed / 10.0)
        obs.append(1.0 if e.shoot_type != "none" else 0.0)
    obs.extend([0.0, 0.0, 0.0, 0.0] * (3 - len(nearest_enemies)))

    # 3 asteroidi piu' vicini (4 valori ciascuno: dx, dy, speed, size)
    nearest_asteroids = _nearest_n(asteroids, px, py, 3)
    for a in nearest_asteroids:
        obs.append((a.rect.centerx - px) / SCREEN_WIDTH)
        obs.append((a.rect.centery - py) / SCREEN_HEIGHT)
        obs.append(a.speed / 10.0)
        obs.append(a.radius / 65.0)
    obs.extend([0.0, 0.0, 0.0, 0.0] * (3 - len(nearest_asteroids)))

    # 3 proiettili nemici piu' vicini (4 valori: dx, dy, vx, vy)
    nearest_bullets = _nearest_n(enemy_bullets, px, py, 3)
    for b in nearest_bullets:
        obs.append((b.rect.centerx - px) / SCREEN_WIDTH)
        obs.append((b.rect.centery - py) / SCREEN_HEIGHT)
        obs.append(b.vx / ENEMY_BULLET_SPEED)
        obs.append(b.vy / ENEMY_BULLET_SPEED)
    obs.extend([0.0, 0.0, 0.0, 0.0] * (3 - len(nearest_bullets)))

    # 1 power-up piu' vicino (2 valori: dx, dy)
    nearest_pu = _nearest_n(powerups, px, py, 1)
    if nearest_pu:
        pu = nearest_pu[0]
        obs.append((pu.rect.centerx - px) / SCREEN_WIDTH)
        obs.append((pu.rect.centery - py) / SCREEN_HEIGHT)
    else:
        obs.extend([0.0, 0.0])

    return obs


class GameEnv:
    """Ambiente di gioco steppabile per AI."""

    def __init__(self, render=False, screen=None):
        self.render_mode = render
        self.screen = screen
        self.frame_count = 0
        self._original_get_ticks = None

        # Gruppi di sprite
        self.bullets = pygame.sprite.Group()
        self.missiles = pygame.sprite.Group()
        self.enemies = pygame.sprite.Group()
        self.asteroids = pygame.sprite.Group()
        self.powerups = pygame.sprite.Group()
        self.explosions = pygame.sprite.Group()
        self.enemy_bullets = pygame.sprite.Group()

        # Giocatore
        self.player = Player()

        # Spawn timers (in frame)
        self.last_enemy_spawn = 0
        self.last_asteroid_spawn = 0
        self.last_powerup_spawn = 0

        # Punteggio e livelli
        self.score = 0
        self.level_manager = LevelManager()

        # Bomba: freeze e campo protettivo (in frame)
        self.bomb_freeze_until = 0
        self.bomb_field_until = 0

        # Statistiche per fitness
        self.lives_lost = 0
        self.powerups_collected = 0
        self.missiles_collected = 0
        self.enemies_killed = 0
        self.missile_kills = 0    # nemici uccisi dai missili
        self.bomb_kills = 0       # nemici distrutti dalle bombe
        self.shield_hits = 0
        self.shield_breaks = 0
        self.idle_frames = 0      # frame in cui la nave non si e' spostata
        self.dodges = 0           # nemici/asteroidi usciti dopo un passaggio ravvicinato
        self.shots_fired = 0
        self.shots_hit = 0
        self.near_misses = 0      # sprite arrivati entro 60px (uno per sprite)
        # Saldo netto dei frame in cui la nave si avvicina (+1) o si allontana
        # (-1) dal power-up piu' vicino: oscillare non accumula nulla
        self.powerup_approach_frames = 0
        self.missiles_fired = 0   # missili lanciati
        self.bombs_used = 0       # bombe fatte esplodere

        # Missile: con l'output alto parte appena consentito, al massimo uno
        # ogni AI_MISSILE_INTERVAL ms (tempo virtuale). Bomba sul fronte di
        # salita dell'azione (come il KEYDOWN umano): valore del frame precedente
        self._next_missile_at = 0
        self._prev_bomb = False
        self._approach_delta = 0

        # Game over
        self.done = False

        # Parallasse e font per rendering
        self._bg = None
        self._hud_font = None
        if self.render_mode:
            from src.parallax import ParallaxBackground
            self._bg = ParallaxBackground()
            self._hud_font = pygame.font.Font(None, 36)

    def _virtual_ticks(self):
        """Ritorna il tempo virtuale in ms basato sul contatore frame."""
        return int(self.frame_count * MS_PER_FRAME)

    def reset(self, seed=None):
        """Reset completo del gioco. Ritorna osservazione iniziale."""
        if seed is not None:
            random.seed(seed)
        self.frame_count = 0
        self.player.lives = PLAYER_LIVES
        self.player.missiles = 0
        self.player.invincible = False
        self.player.invincible_timer = 0
        self.player.visible = True
        self.player.blink_timer = 0
        self.player.reset_position((100, SCREEN_HEIGHT // 2))
        self.player.upgrades.reset()
        self.player._update_sprite(force=True)
        self.player.shield_hp = 0
        self.player.shield_just_broke = False
        self.player.shield_break_timer = 0.0
        self.player.bombs = 0
        # Sparo disponibile da subito, come in Game (dove get_ticks() parte gia'
        # oltre il cooldown): con last_shot = 0 e tempo virtuale 0 i primi
        # shoot_delay ms avrebbero cooldown pieno
        self.player.last_shot = -self.player.upgrades.shoot_delay

        self.bullets.empty()
        self.missiles.empty()
        self.enemies.empty()
        self.asteroids.empty()
        self.powerups.empty()
        self.explosions.empty()
        self.enemy_bullets.empty()

        self.score = 0
        self.level_manager.reset()
        self.bomb_freeze_until = 0
        self.bomb_field_until = 0
        self.lives_lost = 0
        self.powerups_collected = 0
        self.missiles_collected = 0
        self.enemies_killed = 0
        self.missile_kills = 0
        self.bomb_kills = 0
        self.shield_hits = 0
        self.shield_breaks = 0
        self.idle_frames = 0
        self.dodges = 0
        self.shots_fired = 0
        self.shots_hit = 0
        self.near_misses = 0
        self.powerup_approach_frames = 0
        self.missiles_fired = 0
        self.bombs_used = 0
        self._next_missile_at = 0
        self._prev_bomb = False
        self._approach_delta = 0
        self.done = False

        self.last_enemy_spawn = 0
        self.last_asteroid_spawn = 0
        self.last_powerup_spawn = 0

        return self.get_observation()

    def step(self, actions, draw=True):
        """Avanza di 1 frame con le azioni date.

        Args:
            actions: dict con chiavi:
                'vertical': -1 (su), 0, 1 (giu)
                'horizontal': -1 (sx), 0, 1 (dx)
                'shoot': bool
                'missile': bool
                'bomb': bool
            draw: con render attivo, False salta il disegno del frame (lo
                stato avanza comunque, sfondo compreso)

        Returns:
            (observation, reward_delta, done)
        """
        if self.done:
            return self.get_observation(), 0.0, True

        self.frame_count += 1
        dt = 1.0
        prev_score = self.score
        prev_lives = self.player.lives
        prev_kills = self.total_kills
        prev_shield_hits = self.shield_hits
        prev_shield_breaks = self.shield_breaks
        prev_powerups = self.powerups_collected
        prev_missiles = self.missiles_collected
        prev_level = self.level_manager.level
        prev_upgrades = self.total_upgrade_levels

        # Monkey-patch pygame.time.get_ticks per timing deterministico
        self._original_get_ticks = pygame.time.get_ticks
        pygame.time.get_ticks = self._virtual_ticks

        try:
            self._apply_actions(actions, dt)
            self._update(dt)
        finally:
            # Ripristina sempre la funzione originale
            pygame.time.get_ticks = self._original_get_ticks
            self._original_get_ticks = None

        # Calcola reward delta
        lives_delta = prev_lives - self.player.lives
        if lives_delta > 0:
            self.lives_lost += lives_delta

        kills_delta = self.total_kills - prev_kills
        reward = 0.02  # piccolo bonus sopravvivenza
        reward += kills_delta * 5.0  # reward istantaneo per uccisioni (proiettile, missile, bomba)
        reward -= lives_delta * 200
        # Penalita' sparo per evitare spam
        if getattr(self, '_shot_this_frame', False):
            reward -= 0.01

        # Reward istantanei per eventi discreti
        reward -= (self.shield_hits - prev_shield_hits) * 15
        reward -= (self.shield_breaks - prev_shield_breaks) * 60
        reward += (self.powerups_collected - prev_powerups) * 10
        reward += (self.missiles_collected - prev_missiles) * 5
        level_delta = self.level_manager.level - prev_level
        if level_delta > 0:
            reward += self.level_manager.level ** 2 * 5.0
        reward += (self.total_upgrade_levels - prev_upgrades) * 5

        # Near-miss: proiettili/nemici/asteroidi entro 60px, contati una volta
        # per sprite. Il marcatore sta sullo sprite (non un set di id(): CPython
        # riusa gli indirizzi degli oggetti liberati) e serve anche alle schivate
        px = self.player.rect.centerx
        py = self.player.rect.centery
        near_dist = 60
        for group in (self.enemy_bullets, self.enemies, self.asteroids):
            for sprite in group:
                if getattr(sprite, '_near_miss', False):
                    continue
                dx = sprite.rect.centerx - px
                dy = sprite.rect.centery - py
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < near_dist:
                    sprite._near_miss = True
                    self.near_misses += 1
                    reward += 3.0

        # Avvicinamento a power-up: solo lo spostamento della nave in questo
        # frame (misurato in _apply_actions), non lo scorrimento del power-up
        reward += self._approach_delta * 0.1

        # Rendering opzionale
        if draw and self.render_mode and self.screen is not None:
            self._draw()

        return self.get_observation(), reward, self.done

    def _apply_actions(self, actions, dt):
        """Applica le azioni dell'AI al giocatore."""
        speed = self.player.upgrades.player_speed
        dx = actions.get('horizontal', 0) * speed
        dy = actions.get('vertical', 0) * speed

        # Power-up piu' vicino nella sua posizione attuale (scorre dopo, in
        # _update): la distanza prima e dopo il movimento isola il contributo
        # della nave, lo scorrimento del power-up non conta
        rect = self.player.rect
        pos_before = rect.topleft
        target = None
        if self.powerups:
            px, py = rect.center
            target = min(
                self.powerups.sprites(),
                key=lambda p: (p.rect.centerx - px) ** 2 + (p.rect.centery - py) ** 2
            )
            dist_before = (target.rect.centerx - px) ** 2 + (target.rect.centery - py) ** 2

        # Stesso movimento (sub-pixel + clamp) dell'umano e dell'AI in coop
        self.player.move(dx, dy, dt)

        # Inattivita' misurata sullo spostamento reale: anche premere contro
        # il bordo (rect invariato dopo il clamp) conta come fermo
        rect = self.player.rect
        if rect.topleft == pos_before:
            self.idle_frames += 1
        self._approach_delta = 0
        if target is not None:
            px, py = rect.center
            dist_after = (target.rect.centerx - px) ** 2 + (target.rect.centery - py) ** 2
            if dist_after < dist_before:
                self._approach_delta = 1
            elif dist_after > dist_before:
                self._approach_delta = -1
            self.powerup_approach_frames += self._approach_delta

        # Gestione invincibilita'
        if self.player.invincible:
            now = self._virtual_ticks()
            inv_time = self.player.upgrades.invincible_time
            if now - self.player.invincible_timer >= inv_time:
                self.player.invincible = False
                self.player.visible = True
            else:
                self.player.blink_timer += 1
                self.player.visible = (self.player.blink_timer // 3) % 2 == 0

        # Sparo
        if actions.get('shoot', False):
            bullets_before = len(self.bullets)
            self.player.shoot(self.bullets)
            if len(self.bullets) > bullets_before:
                self.shots_fired += 1
                self._shot_this_frame = True
            else:
                self._shot_this_frame = False
        else:
            self._shot_this_frame = False

        # Missile a output tenuto: una rete con l'output sempre alto lancia i
        # missili man mano che li raccoglie, uno ogni AI_MISSILE_INTERVAL ms
        if actions.get('missile', False):
            now = self._virtual_ticks()
            if (now >= self._next_missile_at
                    and self.player.shoot_missile(self.missiles, self.enemies)):
                self._next_missile_at = now + AI_MISSILE_INTERVAL
                self.missiles_fired += 1

        # Bomba sul fronte di salita (come il KEYDOWN umano): tenere alto
        # l'output non svuota le scorte in frame consecutivi
        bomb = bool(actions.get('bomb', False))
        if bomb and not self._prev_bomb and self.player.bombs > 0:
            self._use_bomb()
        self._prev_bomb = bomb

    def _update(self, dt):
        """Aggiorna tutti gli sprite e gestisce le collisioni."""
        lm = self.level_manager

        if self._bg and self.render_mode:
            self._bg.update(dt, lm.level)

        # Update sprite (NON player.update perche' gia' gestito in _apply_actions)
        self.bullets.update(dt)
        # Snapshot: un missile esaurito o uscito dallo schermo lascia il gruppo
        # dentro update(), ma le kill del suo ultimo frame vanno contate
        live_missiles = self.missiles.sprites()
        self.missiles.update(dt)
        # Schivate: nemici e asteroidi che escono dallo schermo (l'unico modo
        # di lasciare il gruppo dentro update(): missili, proiettili e bomba
        # uccidono altrove) dopo un passaggio ravvicinato (near-miss) senza
        # aver colpito la nave; un asteroide che colpisce resta in gioco
        enemies_before = self.enemies.sprites()
        self.enemies.update(dt)
        if len(self.enemies) < len(enemies_before):
            for e in enemies_before:
                if not e.alive() and getattr(e, '_near_miss', False):
                    self.dodges += 1
        asteroids_before = self.asteroids.sprites()
        self.asteroids.update(dt)
        if len(self.asteroids) < len(asteroids_before):
            for a in asteroids_before:
                if (not a.alive() and getattr(a, '_near_miss', False)
                        and not getattr(a, '_hit_player', False)):
                    self.dodges += 1
        self.powerups.update(dt)
        self.explosions.update(dt)
        self.enemy_bullets.update(dt)
        lm.update(dt, FPS)

        now = self._virtual_ticks()
        frozen = self.frame_count < self.bomb_freeze_until

        # Spawn nemici
        enemy_delay_frames = lm.enemy_spawn_delay / MS_PER_FRAME
        if not frozen and (self.frame_count - self.last_enemy_spawn) >= enemy_delay_frames:
            self.last_enemy_spawn = self.frame_count
            shoot_type, shoot_delay = lm.pick_shoot_config()
            self.enemies.add(Enemy(
                speed_min=lm.enemy_speed_min,
                speed_max=lm.enemy_speed_max,
                shoot_type=shoot_type,
                shoot_delay=shoot_delay,
                enemy_bullets=self.enemy_bullets,
                player=self.player,
            ))

        # Spawn asteroidi
        asteroid_delay_frames = lm.asteroid_spawn_delay / MS_PER_FRAME
        if not frozen and (self.frame_count - self.last_asteroid_spawn) >= asteroid_delay_frames:
            self.last_asteroid_spawn = self.frame_count
            self.asteroids.add(Asteroid(
                speed_min=lm.asteroid_speed_min,
                speed_max=lm.asteroid_speed_max,
                size_min=lm.asteroid_size_min,
                size_max=lm.asteroid_size_max,
            ))

        # Spawn power-up missili
        powerup_delay_frames = lm.powerup_spawn_delay / MS_PER_FRAME
        if (self.frame_count - self.last_powerup_spawn) >= powerup_delay_frames:
            self.last_powerup_spawn = self.frame_count
            self.powerups.add(MissilePowerUp())

        # Collisioni proiettile-nemico
        hits = groupcollide_fast(self.bullets, self.enemies, True, True)
        for bullet, enemies_hit in hits.items():
            self.score += len(enemies_hit) * 100
            self.enemies_killed += len(enemies_hit)
            self.shots_hit += 1
            for enemy in enemies_hit:
                self.explosions.add(Explosion(enemy.rect.centerx, enemy.rect.centery))
                self._try_drop_powerup(enemy.rect.centerx, enemy.rect.centery)

        # Punti dai missili (anche quelli usciti dal gruppo in questo frame)
        for missile in live_missiles:
            if missile.total_kills > 0:
                self.score += missile.total_kills * 150
                self.missile_kills += missile.total_kills
                missile.total_kills = 0
            for pos in missile.kill_positions:
                self.explosions.add(Explosion(*pos))
                self._try_drop_powerup(*pos)
            missile.kill_positions.clear()

        # Collisioni proiettile-asteroide
        ast_hits = groupcollide_fast(self.bullets, self.asteroids, True, False)
        for bullet in ast_hits:
            self.explosions.add(Explosion(bullet.rect.centerx, bullet.rect.centery))

        # Raccolta power-up
        collected = pygame.sprite.spritecollide(self.player, self.powerups, True)
        for pu in collected:
            if isinstance(pu, MissilePowerUp):
                self.missiles_collected += 1
                self.player.missiles += 1
            elif isinstance(pu, PowerUp):
                self.powerups_collected += 1
                self.player.collect_powerup(pu.ptype)

        # Level-up
        lm.check_levelup(self.score)

        # Campo protettivo bomba
        bomb_field_active = self.frame_count < self.bomb_field_until

        # Collisione giocatore-nemico
        if not self.player.invincible and not bomb_field_active:
            hit_enemy = pygame.sprite.spritecollideany(self.player, self.enemies)
            if hit_enemy:
                hit_enemy.kill()
                if not self._player_take_hit():
                    return

        # Collisione giocatore-asteroide
        if not self.player.invincible and not bomb_field_active:
            hit_ast = pygame.sprite.spritecollideany(self.player, self.asteroids)
            if hit_ast:
                hit_ast._hit_player = True  # uscendo non conta come schivata
                if not self._player_take_hit():
                    return

        # Collisione proiettile nemico -> giocatore
        if not self.player.invincible and not bomb_field_active:
            eb_hit = pygame.sprite.spritecollideany(
                self.player, self.enemy_bullets
            )
            if eb_hit:
                eb_hit.kill()
                if not self._player_take_hit():
                    return

    def _player_take_hit(self):
        """Gestisce un colpo al player, traccia scudo. Ritorna True se vivo."""
        shield_before = self.player.shield_hp
        alive = self.player.hit()
        if shield_before > 0 and self.player.shield_hp < shield_before:
            self.shield_hits += 1
        if self.player.shield_just_broke:
            self.shield_breaks += 1
        if not alive:
            self.done = True
            return False
        return True

    @property
    def total_upgrade_levels(self):
        return sum(self.player.upgrades.levels.values())

    @property
    def total_kills(self):
        """Nemici uccisi con qualsiasi arma (proiettili, missili, bomba)."""
        return self.enemies_killed + self.missile_kills + self.bomb_kills

    def _try_drop_powerup(self, x, y):
        if random.random() < POWERUP_DROP_CHANCE:
            self.powerups.add(PowerUp(x, y))

    def _use_bomb(self):
        if not self.player.use_bomb():
            return
        self.bombs_used += 1
        up = self.player.upgrades
        score_mult = 2 if up.bomb_double_score else 1

        for enemy in list(self.enemies):
            self.score += 100 * score_mult
            self.bomb_kills += 1
            self.explosions.add(Explosion(enemy.rect.centerx, enemy.rect.centery))
            enemy.kill()

        self.enemy_bullets.empty()

        if up.bomb_destroys_asteroids:
            for ast in list(self.asteroids):
                self.explosions.add(Explosion(ast.rect.centerx, ast.rect.centery))
                ast.kill()

        if up.bomb_freeze_spawn:
            self.bomb_freeze_until = self.frame_count + int(BOMB_FREEZE_DURATION / MS_PER_FRAME)

        if up.bomb_shield_field:
            self.bomb_field_until = self.frame_count + int(BOMB_FIELD_DURATION / MS_PER_FRAME)

    def get_observation(self):
        """Estrae 52 valori normalizzati dallo stato corrente."""
        return build_observation(
            self.player, self.enemies, self.asteroids,
            self.enemy_bullets, self.powerups,
            self.level_manager, self._virtual_ticks()
        )

    def _draw(self):
        """Renderizza il frame corrente (solo se render_mode=True)."""
        if self._bg:
            self._bg.draw(self.screen)
        else:
            self.screen.fill((10, 10, 30))

        self.bullets.draw(self.screen)
        self.missiles.draw(self.screen)
        self.enemy_bullets.draw(self.screen)
        self.powerups.draw(self.screen)
        self.enemies.draw(self.screen)
        self.asteroids.draw(self.screen)
        self.explosions.draw(self.screen)

        if self.player.visible:
            self.screen.blit(self.player.image, self.player.rect)

        # HUD minimale
        font = self._hud_font

        # Vite
        lives_text = font.render(f"Lives: {self.player.lives}", True, (255, 255, 255))
        self.screen.blit(lives_text, (10, 10))

        # Score
        score_text = font.render(f"{self.score}", True, (255, 255, 255))
        score_rect = score_text.get_rect(topright=(SCREEN_WIDTH - 15, 10))
        self.screen.blit(score_text, score_rect)

        # Livello
        level_text = font.render(
            f"Level {self.level_manager.level}", True, (255, 255, 0)
        )
        self.screen.blit(
            level_text, level_text.get_rect(midtop=(SCREEN_WIDTH // 2, 10))
        )

    def close(self):
        """Pulizia risorse."""
        self.bullets.empty()
        self.missiles.empty()
        self.enemies.empty()
        self.asteroids.empty()
        self.powerups.empty()
        self.explosions.empty()
        self.enemy_bullets.empty()
