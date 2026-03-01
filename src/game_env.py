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

MS_PER_FRAME = 1000.0 / FPS  # ~33.3 ms


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
        self.shield_hits = 0
        self.shield_breaks = 0
        self.idle_frames = 0
        self.dodges = 0           # nemici/asteroidi passati senza collisione
        self.shots_fired = 0
        self.shots_hit = 0
        self.near_misses = 0      # schivate ravvicinate (entro 60px)
        self._near_miss_ids = set()  # sprite gia' contati
        self.powerup_approach_frames = 0  # frame in avvicinamento a powerup

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
        self.player.visible = True
        self.player.rect.center = (100, SCREEN_HEIGHT // 2)
        self.player.upgrades.reset()
        self.player._update_sprite(force=True)
        self.player.shield_hp = 0
        self.player.bombs = 0
        self.player.last_shot = 0

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
        self.shield_hits = 0
        self.shield_breaks = 0
        self.idle_frames = 0
        self.dodges = 0
        self.shots_fired = 0
        self.shots_hit = 0
        self.near_misses = 0
        self._near_miss_ids = set()
        self.powerup_approach_frames = 0
        self._prev_powerup_dist = None
        self.done = False

        self.last_enemy_spawn = 0
        self.last_asteroid_spawn = 0
        self.last_powerup_spawn = 0

        return self.get_observation()

    def step(self, actions):
        """Avanza di 1 frame con le azioni date.

        Args:
            actions: dict con chiavi:
                'vertical': -1 (su), 0, 1 (giu)
                'horizontal': -1 (sx), 0, 1 (dx)
                'shoot': bool
                'missile': bool
                'bomb': bool

        Returns:
            (observation, reward_delta, done)
        """
        if self.done:
            return self.get_observation(), 0.0, True

        self.frame_count += 1
        dt = 1.0
        prev_score = self.score
        prev_lives = self.player.lives

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
        score_delta = self.score - prev_score
        lives_delta = prev_lives - self.player.lives
        if lives_delta > 0:
            self.lives_lost += lives_delta

        reward = score_delta + 0.02  # piccolo bonus sopravvivenza
        reward -= lives_delta * 200

        # Near-miss: proiettili/nemici/asteroidi entro 60px senza collisione
        px = self.player.rect.centerx
        py = self.player.rect.centery
        near_dist = 60
        for group in (self.enemy_bullets, self.enemies, self.asteroids):
            for sprite in group:
                sid = id(sprite)
                if sid in self._near_miss_ids:
                    continue
                dx = sprite.rect.centerx - px
                dy = sprite.rect.centery - py
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < near_dist:
                    self._near_miss_ids.add(sid)
                    self.near_misses += 1
                    reward += 2.0

        # Pulizia periodica near-miss IDs (rimuovi sprite non piu' in gioco)
        if self.frame_count % 300 == 0 and self._near_miss_ids:
            alive_ids = set()
            for group in (self.enemy_bullets, self.enemies, self.asteroids):
                for sprite in group:
                    alive_ids.add(id(sprite))
            self._near_miss_ids &= alive_ids

        # Bonus posizione avanzata (non abbracciare bordo sinistro)
        if px > 200:
            reward += 0.01

        # Avvicinamento a powerup
        if self.powerups:
            nearest = min(
                self.powerups.sprites(),
                key=lambda p: (p.rect.centerx - px) ** 2 + (p.rect.centery - py) ** 2
            )
            dist = ((nearest.rect.centerx - px) ** 2 + (nearest.rect.centery - py) ** 2) ** 0.5
            if self._prev_powerup_dist is not None and dist < self._prev_powerup_dist:
                self.powerup_approach_frames += 1
                reward += 0.5
            self._prev_powerup_dist = dist
        else:
            self._prev_powerup_dist = None

        # Rendering opzionale
        if self.render_mode and self.screen is not None:
            self._draw()

        return self.get_observation(), reward, self.done

    def _apply_actions(self, actions, dt):
        """Applica le azioni dell'AI al giocatore."""
        # Conteggio frame senza movimento
        if actions.get('vertical', 0) == 0 and actions.get('horizontal', 0) == 0:
            self.idle_frames += 1

        speed = self.player.upgrades.player_speed
        dx = actions.get('horizontal', 0) * speed
        dy = actions.get('vertical', 0) * speed

        self.player.rect.x += int(dx * dt)
        self.player.rect.y += int(dy * dt)
        self.player.rect.clamp_ip(pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))

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

        # Missile
        if actions.get('missile', False):
            self.player.shoot_missile(self.missiles, self.enemies)

        # Bomba (controllata dall'AI)
        if actions.get('bomb', False) and self.player.bombs > 0:
            self._use_bomb()

    def _update(self, dt):
        """Aggiorna tutti gli sprite e gestisce le collisioni."""
        lm = self.level_manager

        if self._bg and self.render_mode:
            self._bg.update(dt)

        # Update sprite (NON player.update perche' gia' gestito in _apply_actions)
        self.bullets.update(dt)
        self.missiles.update(dt)
        # Traccia nemici/asteroidi che escono dallo schermo (schivate)
        enemies_before = len(self.enemies)
        self.enemies.update(dt)
        asteroids_before = len(self.asteroids)
        self.asteroids.update(dt)
        # Nemici/asteroidi usciti dallo schermo = schivati (non uccisi)
        enemies_exited = enemies_before - len(self.enemies)
        asteroids_exited = asteroids_before - len(self.asteroids)
        # Nota: le uccisioni vengono sottratte dopo nelle collisioni
        # ma qui i kill da proiettile non sono ancora avvenuti
        self.dodges += enemies_exited + asteroids_exited
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
        hits = pygame.sprite.groupcollide(self.bullets, self.enemies, True, True)
        for bullet, enemies_hit in hits.items():
            self.score += len(enemies_hit) * 100
            self.enemies_killed += len(enemies_hit)
            self.shots_hit += 1
            for enemy in enemies_hit:
                self.explosions.add(Explosion(enemy.rect.centerx, enemy.rect.centery))
                self._try_drop_powerup(enemy.rect.centerx, enemy.rect.centery)

        # Punti dai missili
        for missile in self.missiles:
            if missile.total_kills > 0:
                self.score += missile.total_kills * 150
                missile.total_kills = 0
            for pos in missile.kill_positions:
                self.explosions.add(Explosion(*pos))
                self._try_drop_powerup(*pos)
            missile.kill_positions.clear()

        # Collisioni proiettile-asteroide
        ast_hits = pygame.sprite.groupcollide(self.bullets, self.asteroids, True, False)
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

    def _try_drop_powerup(self, x, y):
        if random.random() < POWERUP_DROP_CHANCE:
            self.powerups.add(PowerUp(x, y))

    def _use_bomb(self):
        if not self.player.use_bomb():
            return
        up = self.player.upgrades
        score_mult = 2 if up.bomb_double_score else 1

        for enemy in list(self.enemies):
            self.score += 100 * score_mult
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
