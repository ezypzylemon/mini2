import pygame

class TetrisGame:
    def __init__(self):
        self.width, self.height = 400, 600
        self.block_size = 20
        self.running = True
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Tetris Game")
        self.clock = pygame.time.Clock()

    def start(self):
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(30)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def update(self):
        # 블록 업데이트 로직 추가 예정
        pass

    def render(self):
        self.screen.fill((0, 0, 0))  # 검은 배경
        pygame.display.flip()

if __name__ == "__main__":
    game = TetrisGame()
    game.start()
