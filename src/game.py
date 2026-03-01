import math
import random
import pygame
from settings import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FPS, TITLE,
    WHITE, YELLOW, RED,
    PLAYER_LIVES, LEVEL_MAX,
    POWERUP_DROP_CHANCE,
    BOMB_FREEZE_DURATION, BOMB_FIELD_DURATION,
    ENEMY_BULLET_SPEED,
)
from src.parallax import ParallaxBackground
from src.player import Player
from src.enemy import Enemy
from src.asteroid import Asteroid
from src.missile import MissilePowerUp
from src.explosion import Explosion
from src.enemy_bullet import EnemyBullet
from src.scores import load_scores, save_score
from src.level import LevelManager
from src.powerup import PowerUp
from src.sounds import SoundManager
from src.game_env import build_observation


class Game:
    def __init__(self, screen=None, record_for_ai=False, coop_ai=False):
        if screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption(TITLE)
        else:
            self.screen = screen
        self.clock = pygame.time.Clock()
        self.running = True
        self.game_over = False
        self.game_over_phase = 0  # 0=gioco, 1=inserimento nome, 2=classifica
        self.player_name = ""
        self.high_scores: list[dict] = []
        self.player_rank = -1
        self.cursor_blink = 0

        # Parallasse
        self.background = ParallaxBackground()

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

        # Spawn timers
        now = pygame.time.get_ticks()
        self.last_enemy_spawn = now
        self.last_asteroid_spawn = now
        self.last_powerup_spawn = now

        # Punteggio e livelli
        self.score = 0
        self.level_manager = LevelManager()
        self.font = pygame.font.Font(None, 36)
        self.font_big = pygame.font.Font(None, 72)

        # Bomba: freeze spawn e campo protettivo
        self.bomb_freeze_until = 0  # timestamp ms
        self.bomb_field_until = 0   # timestamp ms

        # Notifica raccolta power-up (testo, colore, timer)
        self.powerup_msg = ""
        self.powerup_msg_color = WHITE
        self.powerup_msg_timer = 0.0

        # Audio
        self.sound = SoundManager()

        # Icone HUD
        self.life_icon = self._create_life_icon()
        self.missile_icon = self._create_missile_icon()
        self.bomb_icon = self._create_bomb_icon()

        # Registrazione sessione per AI
        self.recorder = None
        self._frame_missile_pressed = False  # flag keydown per recorder
        self._frame_bomb_pressed = False     # flag keydown per recorder
        if record_for_ai:
            from src.human_recorder import HumanRecorder
            self.recorder = HumanRecorder()

        # Modalita' coop con AI
        self.coop_ai = coop_ai
        self.ai_player = None
        self.ai_net = None
        self.ai_bullets = pygame.sprite.Group()
        self.ai_missiles = pygame.sprite.Group()
        self.ai_dead_msg_timer = 0.0
        if coop_ai:
            self.ai_player = Player()
            self.ai_player.rect.center = (100, SCREEN_HEIGHT // 2 + 80)
            self._load_ai_genome()

    def _create_life_icon(self) -> pygame.Surface:
        """Piccola icona jet per l'HUD delle vite."""
        surf = pygame.Surface((24, 10), pygame.SRCALPHA)
        c = (120, 130, 150)
        pygame.draw.rect(surf, c, (2, 3, 18, 4))
        pygame.draw.polygon(surf, c, [(20, 5), (24, 5), (20, 3)])
        pygame.draw.polygon(surf, (85, 95, 115), [(8, 3), (12, 0), (14, 3)])
        pygame.draw.polygon(surf, (85, 95, 115), [(8, 7), (12, 10), (14, 7)])
        pygame.draw.line(surf, (85, 95, 115), (3, 2), (5, 3), 1)
        pygame.draw.line(surf, (85, 95, 115), (3, 8), (5, 7), 1)
        return surf

    def _create_missile_icon(self) -> pygame.Surface:
        """Piccola icona missile per l'HUD."""
        surf = pygame.Surface((20, 8), pygame.SRCALPHA)
        # Corpo
        pygame.draw.rect(surf, (200, 60, 40), (4, 2, 12, 4))
        # Ogiva
        pygame.draw.polygon(surf, (220, 220, 220), [(16, 2), (20, 4), (16, 6)])
        # Alette
        pygame.draw.line(surf, (160, 45, 30), (4, 0), (4, 2), 2)
        pygame.draw.line(surf, (160, 45, 30), (4, 6), (4, 8), 2)
        # Fiamma
        pygame.draw.rect(surf, (255, 180, 50), (0, 3, 4, 2))
        return surf

    def _create_bomb_icon(self) -> pygame.Surface:
        """Piccola icona bomba per l'HUD."""
        surf = pygame.Surface((16, 16), pygame.SRCALPHA)
        # Corpo bomba
        pygame.draw.circle(surf, (200, 80, 255), (8, 9), 6)
        pygame.draw.circle(surf, (160, 50, 220), (8, 9), 6, 1)
        # Miccia
        pygame.draw.line(surf, (180, 180, 180), (8, 3), (10, 1), 2)
        # Scintilla
        pygame.draw.circle(surf, (255, 255, 100), (10, 1), 2)
        return surf

    def run(self):
        self.sound.start_music()
        while self.running:
            dt = self.clock.tick(FPS) / (1000.0 / FPS)
            self._handle_events()
            if not self.game_over:
                self._update(dt)
            self._draw()

    def _handle_events(self):
        self._frame_missile_pressed = False
        self._frame_bomb_pressed = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

                # ── Fase 1: inserimento nome ──
                elif self.game_over_phase == 1:
                    if event.key == pygame.K_RETURN and self.player_name:
                        self.player_rank = save_score(
                            self.player_name, self.score
                        )
                        self.high_scores = load_scores()
                        self.game_over_phase = 2
                    elif event.key == pygame.K_BACKSPACE:
                        self.player_name = self.player_name[:-1]
                    else:
                        ch = event.unicode
                        if ch and ch.isprintable() and len(self.player_name) < 10:
                            self.player_name += ch

                # ── Fase 2: classifica → restart ──
                elif self.game_over_phase == 2:
                    if event.key == pygame.K_RETURN:
                        self._restart()

                # ── Gioco attivo ──
                elif not self.game_over:
                    if event.key == pygame.K_m:
                        self._frame_missile_pressed = True
                        if self.player.shoot_missile(self.missiles, self.enemies):
                            self.sound.play_missile()
                    elif event.key == pygame.K_b:
                        self._frame_bomb_pressed = True
                        self._use_bomb()

        if not self.game_over:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                if self.player.shoot(self.bullets):
                    self.sound.play_shoot()

    def _update(self, dt: float):
        # Registra frame per AI (prima di aggiornare lo stato)
        if self.recorder is not None:
            self.recorder.record_frame(self)

        lm = self.level_manager

        self.background.update(dt)
        self.player.update(dt)
        self.bullets.update(dt)
        self.missiles.update(dt)
        self.enemies.update(dt)
        self.asteroids.update(dt)
        self.powerups.update(dt)
        self.explosions.update(dt)
        self.enemy_bullets.update(dt)
        lm.update(dt, FPS)

        now = pygame.time.get_ticks()
        frozen = now < self.bomb_freeze_until

        # Spawn nemici (parametri dal LevelManager)
        if not frozen and now - self.last_enemy_spawn >= lm.enemy_spawn_delay:
            self.last_enemy_spawn = now
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
        if not frozen and now - self.last_asteroid_spawn >= lm.asteroid_spawn_delay:
            self.last_asteroid_spawn = now
            self.asteroids.add(Asteroid(
                speed_min=lm.asteroid_speed_min,
                speed_max=lm.asteroid_speed_max,
                size_min=lm.asteroid_size_min,
                size_max=lm.asteroid_size_max,
            ))

        # Spawn power-up missili (timer dal LevelManager)
        if now - self.last_powerup_spawn >= lm.powerup_spawn_delay:
            self.last_powerup_spawn = now
            self.powerups.add(MissilePowerUp())

        # Aggiorna timer notifica power-up
        if self.powerup_msg_timer > 0:
            self.powerup_msg_timer -= dt / FPS

        # Aggiorna timer effetto rottura scudo
        if self.player.shield_break_timer > 0:
            self.player.shield_break_timer -= dt / FPS

        # Collisioni proiettile-nemico
        hits = pygame.sprite.groupcollide(self.bullets, self.enemies, True, True)
        for bullet, enemies_hit in hits.items():
            self.score += len(enemies_hit) * 100
            for enemy in enemies_hit:
                self.explosions.add(Explosion(enemy.rect.centerx, enemy.rect.centery))
                self._try_drop_powerup(enemy.rect.centerx, enemy.rect.centery)
                self.sound.play_explosion()

        # Punti + esplosioni dai missili (150 punti per kill da missile)
        for missile in self.missiles:
            if missile.total_kills > 0:
                self.score += missile.total_kills * 150
                missile.total_kills = 0
            for pos in missile.kill_positions:
                self.explosions.add(Explosion(*pos))
                self._try_drop_powerup(*pos)
                self.sound.play_explosion()
            missile.kill_positions.clear()

        # Collisioni proiettile-asteroide (scintilla sull'impatto)
        ast_hits = pygame.sprite.groupcollide(self.bullets, self.asteroids, True, False)
        for bullet in ast_hits:
            self.explosions.add(Explosion(bullet.rect.centerx, bullet.rect.centery))

        # Raccolta power-up
        collected = pygame.sprite.spritecollide(self.player, self.powerups, True)
        for pu in collected:
            self.sound.play_powerup()
            if isinstance(pu, MissilePowerUp):
                self.player.missiles += 1
            elif isinstance(pu, PowerUp):
                self.player.collect_powerup(pu.ptype)
                self._show_powerup_msg(pu.ptype)

        # Controllo level-up
        if lm.check_levelup(self.score):
            self.sound.play_level_up()

        # Campo protettivo da bomba lv10
        bomb_field_active = now < self.bomb_field_until

        # Collisione giocatore-nemico
        if not self.player.invincible and not bomb_field_active:
            hit_enemy = pygame.sprite.spritecollideany(self.player, self.enemies)
            if hit_enemy:
                hit_enemy.kill()
                if not self.player.hit():
                    self._trigger_game_over()
                else:
                    self._play_hit_sound()

        # Collisione giocatore-asteroide
        if not self.player.invincible and not bomb_field_active:
            hit_ast = pygame.sprite.spritecollideany(self.player, self.asteroids)
            if hit_ast:
                if not self.player.hit():
                    self._trigger_game_over()
                else:
                    self._play_hit_sound()

        # Collisione proiettile nemico → giocatore
        if not self.player.invincible and not bomb_field_active:
            eb_hit = pygame.sprite.spritecollideany(
                self.player, self.enemy_bullets
            )
            if eb_hit:
                eb_hit.kill()
                if not self.player.hit():
                    self._trigger_game_over()
                else:
                    self._play_hit_sound()

        # Aggiorna AI coop
        if self.coop_ai:
            self._update_ai(dt)
            # Timer messaggio AI eliminata
            if self.ai_dead_msg_timer > 0:
                self.ai_dead_msg_timer -= dt / FPS

    def _play_hit_sound(self):
        """Suono differenziato: scudo assorbimento / scudo rotto / vita persa."""
        if self.player.shield_just_broke:
            self.sound.play_shield_break()
        elif self.player.shield_hp > 0:
            # Lo scudo ha assorbito ma non si è rotto
            self.sound.play_shield_hit()
        else:
            self.sound.play_player_hit()

    def _draw(self):
        self.background.draw(self.screen)

        # Proiettili e missili
        self.bullets.draw(self.screen)
        self.missiles.draw(self.screen)
        # Proiettili AI
        if self.coop_ai:
            self.ai_bullets.draw(self.screen)
            self.ai_missiles.draw(self.screen)
        # Proiettili nemici
        self.enemy_bullets.draw(self.screen)
        # Power-ups
        self.powerups.draw(self.screen)
        # Nemici
        self.enemies.draw(self.screen)
        # Asteroidi
        self.asteroids.draw(self.screen)
        # Esplosioni (sopra tutto tranne HUD)
        self.explosions.draw(self.screen)

        # Nave AI (con tinta azzurra)
        if self.coop_ai and self.ai_player and self.ai_player.visible and self.ai_player.lives > 0:
            tinted = self._tint_surface(self.ai_player.image, (0, 150, 255))
            self.screen.blit(tinted, self.ai_player.rect)
            # Label "AI" sopra la nave
            ai_label = self.font.render("AI", True, (0, 200, 255))
            self.screen.blit(ai_label, ai_label.get_rect(
                midbottom=(self.ai_player.rect.centerx, self.ai_player.rect.top - 2)
            ))

        # Giocatore
        if self.player.visible:
            self.screen.blit(self.player.image, self.player.rect)

        # Bolla scudo attorno al giocatore
        if self.player.shield_hp > 0 and self.player.visible:
            cx, cy = self.player.rect.center
            radius = max(self.player.rect.width, self.player.rect.height) // 2 + 8
            shield_surf = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
            sc = shield_surf.get_rect().center
            # Intensità basata sugli HP (più HP = più opaco)
            intensity = self.player.shield_hp / max(self.player.upgrades.shield_max, 1)
            base_alpha = int(25 + 35 * intensity)
            # Pulsazione leggera
            pulse = 0.85 + 0.15 * math.sin(pygame.time.get_ticks() * 0.005)
            # Alone esterno
            pygame.draw.circle(shield_surf, (80, 150, 255, int(base_alpha * 0.5 * pulse)),
                               sc, radius)
            # Bordo luminoso
            border_alpha = int((60 + 40 * intensity) * pulse)
            pygame.draw.circle(shield_surf, (120, 200, 255, border_alpha),
                               sc, radius, 2)
            self.screen.blit(shield_surf, (cx - sc[0], cy - sc[1]))

        # Effetto rottura scudo (frammenti che si espandono)
        if self.player.shield_break_timer > 0:
            cx, cy = self.player.rect.center
            progress = 1.0 - self.player.shield_break_timer / 0.6
            alpha = int(255 * (1.0 - progress))
            break_surf = pygame.Surface((200, 200), pygame.SRCALPHA)
            bc = break_surf.get_rect().center
            # Anello che si espande
            ring_r = int(30 + 50 * progress)
            ring_alpha = max(0, alpha)
            pygame.draw.circle(break_surf, (120, 200, 255, ring_alpha),
                               bc, ring_r, 3)
            # Frammenti (archi che si allontanano)
            num_frags = 8
            for j in range(num_frags):
                angle = j * (2 * math.pi / num_frags) + progress * 0.5
                frag_r = int(25 + 60 * progress)
                fx = bc[0] + int(math.cos(angle) * frag_r)
                fy = bc[1] + int(math.sin(angle) * frag_r)
                frag_alpha = max(0, int(alpha * 0.8))
                frag_len = max(2, int(8 * (1.0 - progress)))
                ex = fx + int(math.cos(angle) * frag_len)
                ey = fy + int(math.sin(angle) * frag_len)
                pygame.draw.line(break_surf, (150, 220, 255, frag_alpha),
                                 (fx, fy), (ex, ey), 2)
            self.screen.blit(break_surf, (cx - bc[0], cy - bc[1]))

        # ── HUD ──
        # Vite (alto a sinistra)
        for i in range(self.player.lives):
            self.screen.blit(self.life_icon, (10 + i * 28, 10))

        # Scudo (barretta sotto le vite)
        up = self.player.upgrades
        if up.shield_max > 0:
            bar_x, bar_y = 10, 24
            bar_w = up.shield_max * 14
            # Sfondo barra
            pygame.draw.rect(self.screen, (40, 40, 80), (bar_x, bar_y, bar_w, 6))
            # Riempimento
            fill_w = self.player.shield_hp * 14
            if fill_w > 0:
                pygame.draw.rect(self.screen, (80, 150, 255), (bar_x, bar_y, fill_w, 6))
            pygame.draw.rect(self.screen, (120, 180, 255), (bar_x, bar_y, bar_w, 6), 1)

        # Missili (sotto le vite/scudo)
        hud_y = 34 if up.shield_max > 0 else 30
        if self.player.missiles > 0:
            for i in range(self.player.missiles):
                self.screen.blit(self.missile_icon, (10 + i * 24, hud_y))
            # Label tasto
            m_text = self.font.render("[M]", True, (255, 180, 50))
            self.screen.blit(m_text, (10 + self.player.missiles * 24 + 4, hud_y - 2))

        # Bombe (sotto i missili)
        if self.player.bombs > 0:
            bomb_y = hud_y + (14 if self.player.missiles > 0 else 0)
            for i in range(self.player.bombs):
                self.screen.blit(self.bomb_icon, (10 + i * 20, bomb_y))
            b_text = self.font.render("[B]", True, (200, 80, 255))
            self.screen.blit(b_text, (10 + self.player.bombs * 20 + 4, bomb_y))

        # Livello (alto al centro)
        level_text = self.font.render(
            f"Level {self.level_manager.level}", True, YELLOW
        )
        self.screen.blit(
            level_text, level_text.get_rect(midtop=(SCREEN_WIDTH // 2, 10))
        )

        # Score (alto a destra)
        score_text = self.font.render(f"{self.score}", True, WHITE)
        score_rect = score_text.get_rect(topright=(SCREEN_WIDTH - 15, 10))
        self.screen.blit(score_text, score_rect)

        # Livelli potenziamenti (alto a destra, sotto lo score)
        upgrade_y = 32
        upgrade_info = [
            ("M", up.levels["engine"], (80, 220, 80)),
            ("W", up.levels["weapon"], (255, 150, 50)),
            ("S", up.levels["shield"], (80, 150, 255)),
            ("B", up.levels["bomb"], (200, 80, 255)),
        ]
        for label, lv, color in upgrade_info:
            if lv > 0:
                txt = self.font.render(f"{label}{lv}", True, color)
                txt_rect = txt.get_rect(topright=(SCREEN_WIDTH - 15, upgrade_y))
                self.screen.blit(txt, txt_rect)
                upgrade_y += 22

        # Notifica raccolta power-up
        if self.powerup_msg_timer > 0:
            alpha = min(255, int(self.powerup_msg_timer * 200))
            msg_surf = self.font.render(self.powerup_msg, True, self.powerup_msg_color)
            msg_surf.set_alpha(alpha)
            self.screen.blit(
                msg_surf, msg_surf.get_rect(center=(SCREEN_WIDTH // 2,
                                                     SCREEN_HEIGHT // 2 + 30))
            )

        # Campo protettivo bomba (alone viola)
        now_draw = pygame.time.get_ticks()
        if now_draw < self.bomb_field_until:
            shield_surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            pygame.draw.circle(
                shield_surf, (200, 80, 255, 30),
                self.player.rect.center, 80
            )
            self.screen.blit(shield_surf, (0, 0))

        # Notifica LEVEL UP
        if self.level_manager.levelup_timer > 0:
            alpha = min(255, int(self.level_manager.levelup_timer * 200))
            lu_surf = self.font_big.render(
                f"LEVEL {self.level_manager.level}!", True, YELLOW
            )
            lu_surf.set_alpha(alpha)
            self.screen.blit(
                lu_surf, lu_surf.get_rect(center=(SCREEN_WIDTH // 2,
                                                   SCREEN_HEIGHT // 2 - 40))
            )

        # HUD AI coop
        if self.coop_ai:
            if self.ai_player and self.ai_player.lives > 0:
                # Vite AI (in basso a sinistra)
                ai_hud = self.font.render("AI:", True, (0, 200, 255))
                self.screen.blit(ai_hud, (10, SCREEN_HEIGHT - 36))
                for i in range(self.ai_player.lives):
                    tinted_icon = self._tint_surface(self.life_icon, (0, 150, 255))
                    self.screen.blit(tinted_icon, (50 + i * 28, SCREEN_HEIGHT - 32))
            elif self.ai_dead_msg_timer > 0:
                alpha = min(255, int(self.ai_dead_msg_timer * 150))
                msg = self.font.render("AI eliminata!", True, (255, 100, 100))
                msg.set_alpha(alpha)
                self.screen.blit(msg, msg.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 40)))

        # Game Over
        if self.game_over:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            self.screen.blit(overlay, (0, 0))

            cx = SCREEN_WIDTH // 2

            if self.game_over_phase == 1:
                # ── Fase 1: inserimento nome ──
                go_text = self.font_big.render("GAME OVER", True, RED)
                self.screen.blit(go_text, go_text.get_rect(center=(cx, 200)))

                final_score = self.font.render(f"Score: {self.score}", True, WHITE)
                self.screen.blit(final_score, final_score.get_rect(center=(cx, 270)))

                prompt = self.font.render("Inserisci il tuo nome:", True, YELLOW)
                self.screen.blit(prompt, prompt.get_rect(center=(cx, 340)))

                # Campo nome con cursore lampeggiante
                self.cursor_blink += 1
                cursor = "|" if (self.cursor_blink // 15) % 2 == 0 else ""
                name_display = self.player_name + cursor
                name_text = self.font_big.render(name_display, True, WHITE)
                self.screen.blit(name_text, name_text.get_rect(center=(cx, 400)))

                hint = self.font.render("Premi INVIO per confermare", True, (150, 150, 150))
                self.screen.blit(hint, hint.get_rect(center=(cx, 470)))

            elif self.game_over_phase == 2:
                # ── Fase 2: classifica Top 10 ──
                title = self.font_big.render("TOP 10", True, YELLOW)
                self.screen.blit(title, title.get_rect(center=(cx, 100)))

                # Intestazione colonne
                header = self.font.render(
                    f"{'#':<4}{'NOME':<14}{'SCORE':>8}", True, (180, 180, 180)
                )
                self.screen.blit(header, header.get_rect(center=(cx, 160)))

                # Righe classifica
                for i, entry in enumerate(self.high_scores):
                    color = YELLOW if i == self.player_rank else WHITE
                    row_text = self.font.render(
                        f"{i + 1:<4}{entry['name']:<14}{entry['score']:>8}",
                        True, color,
                    )
                    y = 200 + i * 36
                    self.screen.blit(row_text, row_text.get_rect(center=(cx, y)))

                # Punteggio giocatore (se fuori top 10)
                if self.player_rank == -1:
                    your = self.font.render(
                        f"Il tuo score: {self.score}", True, (150, 150, 150)
                    )
                    self.screen.blit(your, your.get_rect(center=(cx, 580)))

                restart_text = self.font.render(
                    "Premi INVIO per ricominciare", True, YELLOW
                )
                self.screen.blit(
                    restart_text, restart_text.get_rect(center=(cx, 630))
                )

        pygame.display.flip()

    def _trigger_game_over(self):
        """Attiva il game over con suono e pausa musica."""
        self.game_over = True
        self.game_over_phase = 1
        self.sound.play_game_over()
        self.sound.pause_music()

        # Salva sessione registrata per AI
        if self.recorder is not None:
            self.recorder.save_session(self.score)
            from src.human_recorder import HumanRecorder
            self.recorder = HumanRecorder()  # reset per la prossima partita

    def _try_drop_powerup(self, x: int, y: int):
        """Probabilità di droppare un power-up alla posizione data."""
        if random.random() < POWERUP_DROP_CHANCE:
            self.powerups.add(PowerUp(x, y))

    def _use_bomb(self):
        """Attiva una bomba: distrugge tutto sullo schermo."""
        if not self.player.use_bomb():
            return
        self.sound.play_bomb()
        up = self.player.upgrades
        score_mult = 2 if up.bomb_double_score else 1

        # Distruggi tutti i nemici
        for enemy in list(self.enemies):
            self.score += 100 * score_mult
            self.explosions.add(Explosion(enemy.rect.centerx, enemy.rect.centery))
            enemy.kill()

        # Distruggi proiettili nemici
        self.enemy_bullets.empty()

        # Distruggi asteroidi (lv2+)
        if up.bomb_destroys_asteroids:
            for ast in list(self.asteroids):
                self.explosions.add(Explosion(ast.rect.centerx, ast.rect.centery))
                ast.kill()

        # Freeze spawn (lv8+)
        if up.bomb_freeze_spawn:
            self.bomb_freeze_until = pygame.time.get_ticks() + BOMB_FREEZE_DURATION

        # Campo protettivo (lv10)
        if up.bomb_shield_field:
            self.bomb_field_until = pygame.time.get_ticks() + BOMB_FIELD_DURATION

    _POWERUP_COLORS = {
        "engine": (80, 220, 80),
        "weapon": (255, 150, 50),
        "shield": (80, 150, 255),
        "bomb": (200, 80, 255),
    }
    _POWERUP_NAMES = {
        "engine": "MOTORE",
        "weapon": "ARMA",
        "shield": "SCUDO",
        "bomb": "BOMBA",
    }

    def _show_powerup_msg(self, ptype: str):
        """Mostra notifica di raccolta power-up."""
        lv = self.player.upgrades.levels[ptype]
        name = self._POWERUP_NAMES.get(ptype, ptype.upper())
        self.powerup_msg = f"+{name} LV.{lv}"
        self.powerup_msg_color = self._POWERUP_COLORS.get(ptype, WHITE)
        self.powerup_msg_timer = 2.0

    # ── AI Coop ──

    def _load_ai_genome(self):
        """Carica il miglior genoma NEAT per la nave AI."""
        import pickle
        import neat
        from src.neat_trainer import CONFIG_PATH, BEST_GENOME_PATH
        config = neat.Config(
            neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation, CONFIG_PATH,
        )
        with open(BEST_GENOME_PATH, 'rb') as f:
            genome = pickle.load(f)
        self.ai_net = neat.nn.FeedForwardNetwork.create(genome, config)

    def _get_ai_observation(self):
        """Estrae 52 valori normalizzati per la rete AI (stesso formato di GameEnv)."""
        return build_observation(
            self.ai_player, self.enemies, self.asteroids,
            self.enemy_bullets, self.powerups,
            self.level_manager, pygame.time.get_ticks()
        )

    @staticmethod
    def _decode_ai_actions(output):
        """Converte 5 output della rete in azioni di gioco."""
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

    def _update_ai(self, dt):
        """Aggiorna la nave AI: osservazione, azione, collisioni."""
        if not self.ai_player or self.ai_player.lives <= 0:
            return

        # Genera osservazione e azione
        obs = self._get_ai_observation()
        output = self.ai_net.activate(obs)
        actions = self._decode_ai_actions(output)

        # Movimento
        speed = self.ai_player.upgrades.player_speed
        dx = actions['horizontal'] * speed
        dy = actions['vertical'] * speed
        self.ai_player.rect.x += int(dx * dt)
        self.ai_player.rect.y += int(dy * dt)
        self.ai_player.rect.clamp_ip(pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))

        # Invincibilita'
        if self.ai_player.invincible:
            now = pygame.time.get_ticks()
            inv_time = self.ai_player.upgrades.invincible_time
            if now - self.ai_player.invincible_timer >= inv_time:
                self.ai_player.invincible = False
                self.ai_player.visible = True
            else:
                self.ai_player.blink_timer += 1
                self.ai_player.visible = (self.ai_player.blink_timer // 3) % 2 == 0

        # Sparo
        if actions['shoot']:
            self.ai_player.shoot(self.ai_bullets)

        # Missile
        if actions['missile']:
            self.ai_player.shoot_missile(self.ai_missiles, self.enemies)

        # Bomba (controllata dall'AI)
        if actions.get('bomb', False) and self.ai_player.bombs > 0:
            if self.ai_player.use_bomb():
                up = self.ai_player.upgrades
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

        # Update proiettili e missili AI
        self.ai_bullets.update(dt)
        self.ai_missiles.update(dt)

        # Collisioni proiettili AI vs nemici
        hits = pygame.sprite.groupcollide(self.ai_bullets, self.enemies, True, True)
        for bullet, enemies_hit in hits.items():
            self.score += len(enemies_hit) * 100
            for enemy in enemies_hit:
                self.explosions.add(Explosion(enemy.rect.centerx, enemy.rect.centery))
                self._try_drop_powerup(enemy.rect.centerx, enemy.rect.centery)
                self.sound.play_explosion()

        # Punti dai missili AI
        for missile in self.ai_missiles:
            if missile.total_kills > 0:
                self.score += missile.total_kills * 150
                missile.total_kills = 0
            for pos in missile.kill_positions:
                self.explosions.add(Explosion(*pos))
                self._try_drop_powerup(*pos)
                self.sound.play_explosion()
            missile.kill_positions.clear()

        # Raccolta power-up da AI
        collected = pygame.sprite.spritecollide(self.ai_player, self.powerups, True)
        for pu in collected:
            self.sound.play_powerup()
            if isinstance(pu, MissilePowerUp):
                self.ai_player.missiles += 1
            elif isinstance(pu, PowerUp):
                self.ai_player.collect_powerup(pu.ptype)

        now = pygame.time.get_ticks()
        bomb_field_active = now < self.bomb_field_until

        # Collisione AI vs nemico
        if not self.ai_player.invincible and not bomb_field_active:
            hit_enemy = pygame.sprite.spritecollideany(self.ai_player, self.enemies)
            if hit_enemy:
                hit_enemy.kill()
                if not self.ai_player.hit():
                    self._ai_eliminated()
                    return

        # Collisione AI vs asteroide
        if not self.ai_player.invincible and not bomb_field_active:
            hit_ast = pygame.sprite.spritecollideany(self.ai_player, self.asteroids)
            if hit_ast:
                if not self.ai_player.hit():
                    self._ai_eliminated()
                    return

        # Collisione proiettile nemico vs AI
        if not self.ai_player.invincible and not bomb_field_active:
            eb_hit = pygame.sprite.spritecollideany(self.ai_player, self.enemy_bullets)
            if eb_hit:
                eb_hit.kill()
                if not self.ai_player.hit():
                    self._ai_eliminated()
                    return

    def _ai_eliminated(self):
        """L'AI ha perso tutte le vite."""
        self.ai_player.lives = 0
        self.ai_player.visible = False
        self.ai_bullets.empty()
        self.ai_missiles.empty()
        self.ai_dead_msg_timer = 3.0

    @staticmethod
    def _tint_surface(surface, tint_color):
        """Ritorna una copia della surface con una tinta di colore applicata."""
        tinted = surface.copy()
        overlay = pygame.Surface(tinted.get_size(), pygame.SRCALPHA)
        overlay.fill((*tint_color, 80))
        tinted.blit(overlay, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
        return tinted

    def _restart(self):
        """Ricomincia il gioco."""
        self.player.lives = PLAYER_LIVES
        self.player.missiles = 0
        self.player.invincible = False
        self.player.visible = True
        self.player.rect.center = (100, SCREEN_HEIGHT // 2)
        self.player.upgrades.reset()
        self.player._update_sprite(force=True)
        self.player.shield_hp = 0
        self.player.shield_break_timer = 0.0
        self.player.shield_just_broke = False
        self.player.bombs = 0
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
        self.powerup_msg_timer = 0.0
        self.game_over = False
        self.game_over_phase = 0
        self.player_name = ""
        self.cursor_blink = 0
        now = pygame.time.get_ticks()
        self.last_enemy_spawn = now
        self.last_asteroid_spawn = now
        self.last_powerup_spawn = now
        self.sound.resume_music()

        # Reset AI coop
        if self.coop_ai and self.ai_player:
            self.ai_player.lives = PLAYER_LIVES
            self.ai_player.missiles = 0
            self.ai_player.invincible = False
            self.ai_player.visible = True
            self.ai_player.rect.center = (100, SCREEN_HEIGHT // 2 + 80)
            self.ai_player.upgrades.reset()
            self.ai_player._update_sprite(force=True)
            self.ai_player.shield_hp = 0
            self.ai_player.shield_break_timer = 0.0
            self.ai_player.shield_just_broke = False
            self.ai_player.bombs = 0
            self.ai_bullets.empty()
            self.ai_missiles.empty()
            self.ai_dead_msg_timer = 0.0
