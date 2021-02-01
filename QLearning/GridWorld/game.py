import pygame
from QLearning.GridWorld.grid_env import (
    GOAL_REWARD,
    TIMESTEP_REWARD,
    TRAP_REWARD,
    GridEnv,
)
from QLearning.GridWorld.QAgent import DoubleQAgent, QAgent

from os.path import join

BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
GREY = (100, 100, 100)

GAMMA_STEP = 0.01
EPSILON_STEP = 0.05
ALPHA_STEP = 0.05

TOGGLE_LEARNING = "toggle_learning"
RESET_PLAYER = "reset_player"
RESET_AGENT_Q_VALUES = "reset_agent"
GAMMA_DOWN = "gamma_down"
GAMMA_UP = "gamma_up"
EPSILON_DOWN = "epsilon_down"
EPSILON_UP = "epsilon_up"
ALPHA_DOWN = "alpha_down"
ALPHA_UP = "alpha_up"
DO_STEP = "do_step"
TOGGLE_Q_LABELS = 'toggle_q_labels'

NOOP = 0
LEFT = 1
UP = 2
RIGHT = 3
DOWN = 4


class GridWorld:
    def __init__(self, layout_id=0, screen_size=(800, 800), display_logo=False):
        pygame.init()
        self.display_logo = display_logo
        self.screen_size = screen_size
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.flip()
        pygame.display.set_caption("GridWorld-QLearning")
        self.player_size = 10

        self.env = GridEnv(layout_id)
        state = self.env.get_state()
        self.agent = QAgent(self.env.observation_space.n, self.env.action_space.n)

        self.margins = (
            self.screen_size[0] // (state.shape[0] + 1),
            self.screen_size[1] // (state.shape[1] + 1),
        )

        self.score = 0
        self.auto_learn = False

        self.display_q_labels = True

    def reset(self):
        self.score = 0
        self.env.reset()
        self.agent.reset()
        pass

    def draw_player(self):
        state = self.env.get_state()
        y, x = state.get_player_pos()
        robot_image = pygame.image.load(join('QLearning', 'GridWorld', 'images', 'robot.png'))
        img_x, img_y = robot_image.get_size()
        pos = (int(x) + 1) * self.margins[0] - img_x//2, (int(y) + 1) * self.margins[1] - img_y//2
        self.screen.blit(robot_image, pos)
        # pygame.draw.circle(self.screen, BLUE, pos, self.player_size)
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

        max_q_value = max(q_values)
        for (x, y), q_value in zip(xy_pos, q_values):
            if max_q_value == q_value:
                self.draw_text(x, y, text="{:.3f}".format(q_value), color_text=RED)
            else:
                self.draw_text(x, y, text="{:.3f}".format(q_value))

    def draw_grid(self):
        cell_width = self.margins[0]
        cell_height = self.margins[1]
        state = self.env.get_state()
        for x in range(0, state.shape[0]):
            for y in range(0, state.shape[0]):
                posx = (x + 1) * self.margins[0] - 0.5 * self.margins[0]
                posy = (y + 1) * self.margins[1] - 0.5 * self.margins[1]
                if state.get_state_transpose(x, y) == 1:
                    rect = pygame.Rect(posx, posy, cell_width, cell_height)
                    pygame.draw.rect(self.screen, GREY, rect, 0)
                elif state.get_state_transpose(x, y) == 2:
                    rect = pygame.Rect(posx, posy, cell_width, cell_height)
                    pygame.draw.rect(self.screen, RED, rect, 0)
                    if self.display_q_labels:
                        self.draw_text(
                            posx + cell_width // 2,
                            posy + cell_width // 2,
                            text=str(TRAP_REWARD),
                            )

                elif state.get_state_transpose(x, y) == 3:
                    rect = pygame.Rect(posx, posy, cell_width, cell_height)
                    pygame.draw.rect(self.screen, GREEN, rect, 0)
                    if self.display_q_labels:
                        self.draw_text(
                            posx + cell_width // 2,
                            posy + cell_height // 2,
                            text=str(GOAL_REWARD),
                            )
                else:
                    rect = pygame.Rect(posx, posy, cell_width, cell_height)
                    pygame.draw.rect(self.screen, BLACK, rect, 1)
                    if self.display_q_labels:
                        self.draw_q_labels(
                            posx,
                            posy,
                            width=self.margins[0],
                            q_values=self.agent.q_table[10 * y + x, 1:].tolist(),
                        )

    def draw_agent_params(self):
        epsilon = self.agent.epsilon
        gamma = self.agent.gamma
        alpha = self.agent.alpha
        episode_reward = self.agent.sum_rewards
        return_ = self.agent.return_
        self.draw_text(
            50,
            40,
            pos="topleft",
            text="Epsilon {:.2f}, Gamma {:.2f}, Alpha {:.2f}"
                 ", Episode Reward {:.2f}, Episode Return {:.2f}"
                .format(
                epsilon, gamma, alpha,
                episode_reward, return_
            ),
        )

    def draw_text(
        self,
        x,
        y,
        pos="center",
        text="",
        rotate_degrees=None,
        color_text=BLACK,
        color_background=WHITE,
    ):
        font = pygame.font.Font("freesansbold.ttf", 16)
        text = font.render(text, True, color_text, color_background)
        textRect = text.get_rect()
        if rotate_degrees:
            pygame.transform.rotate(text, rotate_degrees)

        if pos == "center":
            textRect.center = (x, y)
        if pos == "topleft":
            textRect.topleft = (x, y)
        self.screen.blit(text, textRect)

    def draw_logo(self):
        robot_image = pygame.image.load(join('QLearning', 'GridWorld', 'images', 'logo_transparant.png'))
        size = robot_image.get_size()
        size = (int(size[0] / 4.1), int(size[1] / 4.1))
        robot_image = pygame.transform.scale(robot_image, size)
        pos = (self.screen_size[0] // 2 - robot_image.get_size()[0]//2 - 70,
               self.screen_size[1] // 2 - robot_image.get_size()[1]//2 - 50)
        self.screen.blit(robot_image, pos)

    def draw_game(self):
        self.screen.fill((255, 255, 255))
        self.draw_grid()
        self.draw_player()
        self.draw_agent_params()
        if self.display_logo:
            self.display_logo()
        pygame.display.update()

    def get_action_from_input(self):
        return_event = None
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = LEFT
                if event.key == pygame.K_UP:
                    action = UP
                if event.key == pygame.K_RIGHT:
                    action = RIGHT
                if event.key == pygame.K_DOWN:
                    action = DOWN
                if event.key == pygame.K_n:
                    action = NOOP

                if event.key == pygame.K_z:
                    return_event = DO_STEP

                if event.key == pygame.K_SPACE:
                    return_event = TOGGLE_LEARNING

                if event.key == pygame.K_r:
                    return_event = RESET_PLAYER

                if event.key == pygame.K_t:
                    return_event = RESET_AGENT_Q_VALUES

                if event.key == pygame.K_q:
                    return_event = EPSILON_UP
                if event.key == pygame.K_a:
                    return_event = EPSILON_DOWN

                if event.key == pygame.K_w:
                    return_event = GAMMA_UP
                if event.key == pygame.K_s:
                    return_event = GAMMA_DOWN

                if event.key == pygame.K_e:
                    return_event = ALPHA_UP
                if event.key == pygame.K_d:
                    return_event = ALPHA_DOWN
                if event.key == pygame.K_l:
                    return_event = TOGGLE_Q_LABELS

        return return_event, action

    def step(self):
        done = False

        while not done:
            # pygame.time.delay(100)
            done = False

            event, action = self.get_action_from_input()
            obs = self.env.get_obs()
            if event == TOGGLE_LEARNING:
                self.auto_learn = not self.auto_learn

            if event == TOGGLE_Q_LABELS:
                self.display_q_labels = not self.display_q_labels

            if self.auto_learn:
                action = self.agent.compute_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                self.agent.update(
                    obs=obs,
                    action=action,
                    next_obs=next_obs,
                    reward=reward,
                    done=done,
                )
            else:
                if event:
                    if event == RESET_PLAYER:
                        self.reset()
                    if event == RESET_AGENT_Q_VALUES:
                        self.agent.reset_q_table()

                    if event == EPSILON_UP:
                        self.agent.epsilon += EPSILON_STEP
                    if event == EPSILON_DOWN:
                        self.agent.epsilon -= EPSILON_STEP

                    if event == GAMMA_UP:
                        self.agent.gamma += GAMMA_STEP
                    if event == GAMMA_DOWN:
                        self.agent.gamma -= GAMMA_STEP

                    if event == ALPHA_UP:
                        self.agent.alpha += ALPHA_STEP
                    if event == ALPHA_DOWN:
                        self.agent.alpha -= ALPHA_STEP

                    if action == DO_STEP:
                        action = self.agent.compute_action(obs)
                if action:
                    if action >= 0:
                        next_obs, reward, done, info = self.env.step(action)
                        self.agent.update(
                            obs=obs,
                            action=action,
                            next_obs=next_obs,
                            reward=reward,
                            done=done,
                        )
                        print(
                            "Time:",
                            self.env.time,
                            "Score:",
                            self.score,
                            "Obs:",
                            next_obs,
                            "reward: ",
                            reward,
                        )
                        self.score += reward

            self.draw_game()
            if done:
                self.reset()
                done = False
        return done


if __name__ == "__main__":
    env = GridWorld(layout_id=0)
    env.reset()
    done = env.step()
# print(env.agent.q_table)
