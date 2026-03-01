import pygame
from settings import (
    SCREEN_WIDTH, SCREEN_HEIGHT,
    PLAYER_SPEED, PLAYER_SIZE, PLAYER_COLOR,
    PLAYER_SHOOT_DELAY, PLAYER_LIVES, PLAYER_INVINCIBLE_TIME,
)
from src.bullet import Bullet
from src.missile import Missile
from src.upgrades import UpgradeManager
from src.ship_sprites import create_ship_sprite


class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self._weapon_level = 0
        self.image = create_ship_sprite(0)
        self.rect = self.image.get_rect(
            center=(100, SCREEN_HEIGHT // 2)
        )
        self.speed = PLAYER_SPEED
        self.last_shot = 0
        self.shoot_delay = PLAYER_SHOOT_DELAY
        self.lives = PLAYER_LIVES
        self.missiles = 0  # missili bonus raccolti
        self.invincible = False
        self.invincible_timer = 0
        self.visible = True
        self.blink_timer = 0

        # Potenziamenti
        self.upgrades = UpgradeManager()
        self.shield_hp = 0
        self.bombs = 0

        # Effetto rottura scudo
        self.shield_break_timer = 0.0  # secondi rimasti per l'effetto
        self.shield_just_broke = False  # flag per suono differenziato

    def _update_sprite(self, force: bool = False):
        """Aggiorna lo sprite se il livello weapon è cambiato."""
        new_level = self.upgrades.levels["weapon"]
        if force or new_level != self._weapon_level:
            self._weapon_level = new_level
            old_center = self.rect.center
            self.image = create_ship_sprite(new_level)
            self.rect = self.image.get_rect(center=old_center)

    def update(self, dt: float):
        keys = pygame.key.get_pressed()

        # Velocità dal potenziamento motore
        current_speed = self.upgrades.player_speed

        # Movimento su tutti e 4 gli assi
        dx = 0
        dy = 0
        if keys[pygame.K_UP]:
            dy = -current_speed
        if keys[pygame.K_DOWN]:
            dy = current_speed
        if keys[pygame.K_LEFT]:
            dx = -current_speed
        if keys[pygame.K_RIGHT]:
            dx = current_speed

        self.rect.x += int(dx * dt)
        self.rect.y += int(dy * dt)
        self.rect.clamp_ip(pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))

        # Gestione invincibilità (lampeggio)
        if self.invincible:
            now = pygame.time.get_ticks()
            inv_time = self.upgrades.invincible_time
            if now - self.invincible_timer >= inv_time:
                self.invincible = False
                self.visible = True
            else:
                self.blink_timer += 1
                self.visible = (self.blink_timer // 3) % 2 == 0

    def shoot(self, bullet_group: pygame.sprite.Group) -> bool:
        now = pygame.time.get_ticks()
        delay = self.upgrades.shoot_delay
        if now - self.last_shot >= delay:
            self.last_shot = now
            offsets = self.upgrades.get_bullet_offsets()
            for offset_y in offsets:
                bullet = Bullet(self.rect.right, self.rect.centery + offset_y)
                bullet_group.add(bullet)
            return True
        return False

    def shoot_missile(self, missile_group: pygame.sprite.Group,
                      enemies: pygame.sprite.Group) -> bool:
        """Lancia un missile se ne ha almeno uno. Ritorna True se lanciato."""
        if self.missiles <= 0:
            return False
        self.missiles -= 1
        missile = Missile(self.rect.right + 10, self.rect.centery, enemies)
        missile_group.add(missile)
        return True

    def use_bomb(self) -> bool:
        """Usa una bomba se disponibile. Ritorna True se usata."""
        if self.bombs <= 0:
            return False
        self.bombs -= 1
        return True

    def collect_powerup(self, ptype: str):
        """Raccoglie un power-up e aggiorna i livelli."""
        self.upgrades.upgrade(ptype)
        if ptype == "weapon":
            self._update_sprite()
        elif ptype == "shield":
            # Rigenera scudo al massimo del nuovo livello
            self.shield_hp = self.upgrades.shield_max
        elif ptype == "bomb":
            # Aggiunge 1 bomba (fino al max)
            self.bombs = min(self.bombs + 1, self.upgrades.max_bombs)

    def hit(self) -> bool:
        """Chiamato quando il giocatore viene colpito.
        Ritorna True se il giocatore è ancora vivo.
        Setta self.shield_just_broke = True se lo scudo si rompe (hp -> 0)."""
        if self.invincible:
            return True
        self.shield_just_broke = False
        # Lo scudo assorbe il colpo prima delle vite
        if self.shield_hp > 0:
            self.shield_hp -= 1
            if self.shield_hp == 0:
                self.shield_just_broke = True
                self.shield_break_timer = 0.6  # durata effetto rottura
            self.invincible = True
            self.invincible_timer = pygame.time.get_ticks()
            self.blink_timer = 0
            return True
        self.lives -= 1
        if self.lives <= 0:
            return False
        self.invincible = True
        self.invincible_timer = pygame.time.get_ticks()
        self.blink_timer = 0
        return True
