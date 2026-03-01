"""Registra le sessioni umane come coppie (osservazione, azione).

Le sessioni salvate vengono usate dal trainer NEAT per calcolare
un bonus di fitness "imitation": i genomi che si comportano come
il giocatore umano nelle stesse situazioni ricevono un bonus.
"""

import os
import pickle
import time
import pygame
from src.game_env import build_observation


AI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ai")
SESSIONS_DIR = os.path.join(AI_DIR, "human_sessions")


class HumanRecorder:
    """Registra coppie (osservazione, azione) durante il gioco umano."""

    def __init__(self):
        self.frames = []

    def record_frame(self, game):
        """Registra un singolo frame della sessione umana.

        Estrae l'osservazione dallo stato del gioco e le azioni
        dai tasti premuti + flag keydown per missile/bomba.
        """
        obs = self._get_observation(game)
        actions = self._get_actions_from_keys(game)
        self.frames.append((obs, actions))

    @staticmethod
    def _get_observation(game):
        """Estrae lo stesso vettore di osservazione a 52 valori usato da GameEnv."""
        return build_observation(
            game.player, game.enemies, game.asteroids,
            game.enemy_bullets, game.powerups,
            game.level_manager, pygame.time.get_ticks()
        )

    def _get_actions_from_keys(self, game):
        """Converte i tasti premuti nel formato azioni dell'AI.

        Missile e bomba sono eventi keydown, non held keys,
        quindi li leggiamo dai flag settati in Game._handle_events().
        """
        keys = pygame.key.get_pressed()

        vertical = 0
        if keys[pygame.K_UP]:
            vertical = -1
        elif keys[pygame.K_DOWN]:
            vertical = 1

        horizontal = 0
        if keys[pygame.K_LEFT]:
            horizontal = -1
        elif keys[pygame.K_RIGHT]:
            horizontal = 1

        return {
            'vertical': vertical,
            'horizontal': horizontal,
            'shoot': bool(keys[pygame.K_SPACE]),
            'missile': game._frame_missile_pressed,
            'bomb': game._frame_bomb_pressed,
        }

    def save_session(self, score):
        """Salva la sessione registrata su disco."""
        if not self.frames:
            return

        os.makedirs(SESSIONS_DIR, exist_ok=True)

        session_data = {
            'frames': self.frames,
            'score': score,
            'num_frames': len(self.frames),
            'timestamp': time.time(),
        }

        filename = f"session_{int(time.time())}.pkl"
        filepath = os.path.join(SESSIONS_DIR, filename)

        with open(filepath, 'wb') as f:
            pickle.dump(session_data, f)

        # Tieni max 20 sessioni (cancella le piu' vecchie)
        self._cleanup_old_sessions()

    def _cleanup_old_sessions(self, max_sessions=20):
        """Rimuove le sessioni piu' vecchie se ce ne sono troppe."""
        if not os.path.exists(SESSIONS_DIR):
            return

        files = sorted(
            [f for f in os.listdir(SESSIONS_DIR) if f.endswith('.pkl')],
            key=lambda f: os.path.getmtime(os.path.join(SESSIONS_DIR, f))
        )

        while len(files) > max_sessions:
            os.remove(os.path.join(SESSIONS_DIR, files.pop(0)))
