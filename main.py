import pygame
from settings import SCREEN_WIDTH, SCREEN_HEIGHT, TITLE


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(TITLE)

    from src.menu import MainMenu

    while True:
        choice = MainMenu(screen).run()

        if choice == 'quit':
            break
        elif choice == 'play':
            from src.game import Game
            Game(screen=screen, record_for_ai=True).run()
        elif choice == 'train_headless':
            from src.neat_trainer import NeatTrainer
            NeatTrainer(screen=screen).train(headless=True)
        elif choice == 'train_visual':
            from src.neat_trainer import NeatTrainer
            NeatTrainer(screen=screen).train(headless=False)
        elif choice == 'coop':
            from src.game import Game
            Game(screen=screen, record_for_ai=True, coop_ai=True).run()

    pygame.quit()


if __name__ == "__main__":
    main()
