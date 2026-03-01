"""Menu principale di Yellow Star con selezione modalita'."""

import os
import pygame
from settings import SCREEN_WIDTH, SCREEN_HEIGHT

BEST_GENOME_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "ai", "best_genome.pkl"
)


class MainMenu:
    """Menu all'avvio: 4 modalita' di gioco."""

    CHOICES = [
        ("Gioca", "play"),
        ("Allena AI", "train_headless"),
        ("Vedi AI che si allena", "train_visual"),
        ("Gioca con AI", "coop"),
    ]

    def __init__(self, screen):
        self.screen = screen
        self.selected = 0
        self.font_title = pygame.font.Font(None, 80)
        self.font_option = pygame.font.Font(None, 48)
        self.font_hint = pygame.font.Font(None, 28)
        self.has_saved_genome = os.path.exists(BEST_GENOME_PATH)

    def run(self):
        """Esegue il menu. Ritorna 'play', 'train_headless', 'train_visual', 'coop' o 'quit'."""
        clock = pygame.time.Clock()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 'quit'
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.selected = (self.selected - 1) % len(self.CHOICES)
                    elif event.key == pygame.K_DOWN:
                        self.selected = (self.selected + 1) % len(self.CHOICES)
                    elif event.key == pygame.K_RETURN:
                        choice = self.CHOICES[self.selected][1]
                        if choice == 'coop' and not self.has_saved_genome:
                            continue
                        return choice
                    elif event.key == pygame.K_ESCAPE:
                        return 'quit'

            self._draw()
            pygame.display.flip()
            clock.tick(30)

    def _draw(self):
        self.screen.fill((10, 10, 30))

        # Titolo
        title = self.font_title.render("YELLOW STAR", True, (255, 255, 0))
        self.screen.blit(title, title.get_rect(center=(SCREEN_WIDTH // 2, 180)))

        # Sottotitolo
        sub = self.font_hint.render("AI Edition", True, (150, 150, 200))
        self.screen.blit(sub, sub.get_rect(center=(SCREEN_WIDTH // 2, 240)))

        # Opzioni
        for i, (label, key) in enumerate(self.CHOICES):
            if i == self.selected:
                color = (255, 255, 0)
            elif key == 'coop' and not self.has_saved_genome:
                color = (80, 80, 80)
            else:
                color = (180, 180, 180)

            prefix = "> " if i == self.selected else "  "
            text = self.font_option.render(f"{prefix}{label}", True, color)
            y = 310 + i * 60
            self.screen.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, y)))

        # Descrizioni
        descriptions = {
            0: "Gioca con la tastiera (le tue partite insegnano all'AI!)",
            1: "Training veloce in background, vedi solo le statistiche",
            2: "Guarda l'AI giocare e imparare in tempo reale",
            3: "Gioca insieme all'AI: due navi, stesso mondo!"
               if self.has_saved_genome
               else "Prima allena l'AI per almeno qualche generazione!",
        }
        desc = self.font_hint.render(
            descriptions[self.selected], True, (130, 130, 160)
        )
        self.screen.blit(desc, desc.get_rect(center=(SCREEN_WIDTH // 2, 570)))

        # Hint navigazione
        hint = self.font_hint.render(
            "SU/GIU per selezionare, INVIO per confermare, ESC per uscire",
            True, (100, 100, 120)
        )
        self.screen.blit(hint, hint.get_rect(center=(SCREEN_WIDTH // 2, 630)))

        # Stato genoma
        if self.has_saved_genome:
            info = self.font_hint.render(
                "Genoma AI salvato trovato", True, (80, 200, 80)
            )
        else:
            info = self.font_hint.render(
                "Nessun genoma AI salvato", True, (200, 150, 80)
            )
        self.screen.blit(info, info.get_rect(center=(SCREEN_WIDTH // 2, 665)))
