import pygame

class TetrisRenderer:
    def __init__(self, screen):
        self.screen = screen

    def draw_block(self, x, y, color):
        block_size = 20
        pygame.draw.rect(self.screen, color, (x, y, block_size, block_size))
