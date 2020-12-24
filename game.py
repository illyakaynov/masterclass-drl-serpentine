import pygame
from grid_env import GridEnv

BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
GREY = (100, 100, 100)


actions = {"left": 0, "up": 1, "right": 2, "down": 3}


class GridWorld:
    def __init__(self):
        pygame.init()

        self.screen_size = (800, 800)
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.flip()
        pygame.display.set_caption("GridWorld-QLearning")
        self.player_size = 10

        self.env = GridEnv()
        from QAgent import QAgent

        state = self.env.get_state()
        self.agent = QAgent(self.env.observation_space.n, self.env.action_space.n)

        self.margins = (
            self.screen_size[0] // (state.shape[0] + 1),
            self.screen_size[1] // (state.shape[1] + 1),
        )

        self.score = 0

    def reset(self):
        self.score = 0
        self.env.reset()
        pass

    def draw_player(self):
        state = self.env.get_state()
        y, x = state.get_player_pos()
        pos = (int(x) + 1) * self.margins[0], (int(y) + 1) * self.margins[1]
        pygame.draw.circle(self.screen, BLUE, pos, self.player_size)
        # pygame.display.update()

    def draw_q_labels(self, x, y, width, q_values, height=None):
        height = height or width
        offset = 5

        half = width // 2
        small = width // offset
        big = (offset - 1) * width // offset

        xy_pos = [
            [x + small, y + half],  # left
            [x + half, y + small],  # up
            [x + big, y + half],  # right
            [x + half, y + big],  # down
        ]

        for (x, y), q_value in zip(xy_pos, q_values):
            self.draw_text(x, y, text="{:.3f}".format(q_value))

    def draw_grid(self):
        cell_width = self.margins[0]
        state = self.env.get_state()
        for x in range(0, state.shape[0]):
            for y in range(0, state.shape[0]):
                posx = (x + 1) * self.margins[0] - 0.5 * self.margins[0]
                posy = (y + 1) * self.margins[1] - 0.5 * self.margins[0]
                if state.get_state_transpose(x, y) == 1:
                    rect = pygame.Rect(posx, posy, cell_width, cell_width)
                    pygame.draw.rect(self.screen, GREY, rect, 0)
                elif state.get_state_transpose(x, y) == 2:
                    rect = pygame.Rect(posx, posy, cell_width, cell_width)
                    pygame.draw.rect(self.screen, RED, rect, 0)
                elif state.get_state_transpose(x, y) == 3:
                    rect = pygame.Rect(posx, posy, cell_width, cell_width)
                    pygame.draw.rect(self.screen, GREEN, rect, 0)
                else:
                    rect = pygame.Rect(posx, posy, cell_width, cell_width)
                    pygame.draw.rect(self.screen, BLACK, rect, 1)
                    self.draw_q_labels(
                        posx,
                        posy,
                        width=self.margins[0],
                        q_values=self.agent.q_table[10 * y + x, 1:].tolist(),
                    )

    def draw_text(self, x, y, text="", rotate_degrees=None):
        font = pygame.font.Font("freesansbold.ttf", 16)
        text = font.render(text, True, GREEN, BLUE)
        textRect = text.get_rect()
        if rotate_degrees:
            pygame.transform.rotate(text, rotate_degrees)

        textRect.center = (x, y)
        self.screen.blit(text, textRect)

    def draw_game(self):
        self.screen.fill((255, 255, 255))
        self.draw_grid()
        self.draw_player()
        pygame.display.update()

    def get_action_from_input(self):
        action = -1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = 1
                if event.key == pygame.K_UP:
                    action = 2
                if event.key == pygame.K_RIGHT:
                    action = 3
                if event.key == pygame.K_DOWN:
                    action = 4
                if event.key == pygame.K_SPACE:
                    action = -3
                if event.key == pygame.K_r:
                    action = -2
                if event.key == pygame.K_t:
                    action = -4
        return action

    def step(self, action=None):
        # pygame.time.delay(100)
        done = False
        if action is None:
            action = self.get_action_from_input()

        old_obs = self.env.get_obs()
        if action == -2:
            self.reset()
        if action == -3:
            action = self.agent.compute_action(self.env.get_obs())
        if action == -4:
            self.agent.reset()
        if action >= 0:
            obs, reward, done, info = self.env.step(action)
            self.agent.update(
                state=old_obs, action=action, next_state=obs, reward=reward
            )
            print(
                "Time:",
                self.env.time,
                "Score:",
                self.score,
                "Obs:",
                obs,
                "reward: ",
                reward,
            )
            self.score += reward

        self.draw_game()
        return done


env = GridWorld()
env.reset()
done = False
while not done:
    done = env.step()
    print(env.agent.q_table)
    if done:
        env.reset()
        done = False
