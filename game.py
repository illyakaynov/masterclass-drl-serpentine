import sys
import time
from dataclasses import dataclass

import gym
import numpy as np
import pygame
from pygame.locals import *
from state import State

BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
GREY = (100, 100, 100)


actions = {"left": 0, "up": 1, "right": 2, "down": 3}


@dataclass
class GridCell:
    pos_x: int
    pos_y: int


class GridEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.state = State()
        self.time = 0
        self.end_time = 20

    def reset(self):
        self.state.reset()
        self.time = 0

    def apply_action(self, action):
        if action == 1:
            self.state.move(dy=-1)
        elif action == 2:
            self.state.move(dx=-1)
        elif action == 3:
            self.state.move(dy=+1)
        elif action == 4:
            self.state.move(dx=+1)
        elif action == 0:
            pass
        else:
            raise ValueError("Unknown action {}".format(action))

    def get_obs(self):
        x, y = self.state.get_player_pos()
        obs = 10 * x + y
        return obs

    def step(self, action):
        self.apply_action(action)
        reward = self.compute_reward()
        obs = self.get_obs()
        self.time += 1

        return obs, reward, done, {}

    def compute_reward(self):
        x, y = self.state.get_player_pos()
        cell_value = self.state.get_state(x, y)
        reward = 0

        if cell_value == 2:
            reward -= 10
        elif cell_value == 3:
            reward += 50
        else:
            reward -= 1

        return reward

    def get_state(self):
        return self.state


class GridWorld:
    def __init__(self):
        pygame.init()

        self.screen_size = (800, 800)
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.flip()
        pygame.display.set_caption("GridWorld-QLearning")
        self.player_size = 10

        self.env = GridEnv()
        state = self.env.get_state()

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
        xy_pos = [
            [
                x + width // 2,
                y + width // offset,
            ],
            [x + width // 2, y + (offset - 1) * width // offset],
            [x + width // offset, y + width // 2],
            [x + (offset - 1) * width // offset, y + width // 2],
        ]

        for (x, y), q_value in zip(xy_pos, q_values):
            self.draw_text(x, y, text=str(q_value))

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
                        posx, posy, width=self.margins[0], q_values=[0.1, 0.2, 0.3, 0.5]
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
                    action = 0
                if event.key == pygame.K_r:
                    action = -2
        return action

    def step(self, action=None):
        # pygame.time.delay(100)
        if action is None:
            action = self.get_action_from_input()

        if action == -2:
            self.reset()
        if action >= 0:
            obs, reward, done, info = self.env.step(action)
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


env = GridWorld()
env.reset()
done = False
score = 0
while not done:
    env.step()
