import gym
from gym import spaces

from QLearning.GridWorld.state import State

TRAP_REWARD = -10
GOAL_REWARD = 50
TIMESTEP_REWARD = -0.1

class GridEnv(gym.Env):
    def __init__(self, layout_id=0):
        super().__init__()
        self.state = State(layout_id=layout_id)
        self.time = 0
        self.end_time = 20
        max_state = 10 * self.state.shape[0] + self.state.shape[1]
        self.observation_space = spaces.Discrete(max_state)
        self.action_space = spaces.Discrete(5)

    def reset(self):
        self.state.reset()
        self.time = 0
        return self.get_obs()

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
        done = False

        self.apply_action(action)
        reward = self.compute_reward()
        obs = self.get_obs()
        self.time += 1

        x, y = self.state.get_player_pos()
        if self.state.get_state(x, y) == 2:
            done = True
        if self.state.get_state(x, y) == 3:
            done = True

        if self.time >= self.end_time:
            done = True
        return obs, reward, done, {}

    def compute_reward(self):
        x, y = self.state.get_player_pos()
        cell_value = self.state.get_state(x, y)
        reward = 0

        if cell_value == 2:
            reward += TRAP_REWARD
        elif cell_value == 3:
            reward += GOAL_REWARD
        else:
            reward += TIMESTEP_REWARD

        return reward

    def get_state(self):
        return self.state

    # def render(self, mode='txt'):
    #     for
    #
    #     self.state.render()
