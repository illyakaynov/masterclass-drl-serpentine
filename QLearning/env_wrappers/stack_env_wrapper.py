import gym
import numpy as np
from gym.spaces.box import Box

import matplotlib.pyplot as plt

from QLearning.env_wrappers.preproccessing_wrapper import AtariPreprocessing


def plot_stack(frame_stack):
    num_frames = frame_stack.shape[-1]
    fig, ax = plt.subplots(1, num_frames, figsize=(20, 20), )
    for i in range(num_frames):
        ax.flat[i].imshow(frame_stack[..., i], cmap='gray')
        ax.flat[i].grid(False)
    plt.show()


class AtariFrameStack(AtariPreprocessing):
    """A class implementing image preprocessing for Atari 2600 agents.

    Specifically, this provides the following subset from the JAIR paper
    (Bellemare et al., 2013) and Nature DQN paper (Mnih et al., 2015):

      * Frame skipping (defaults to 4).
      * Terminal signal when a life is lost (off by default).
      * Grayscale and max-pooling of the last two frames.
      * Downsample the screen to a square image (defaults to 84x84).

    More generally, this class follows the preprocessing guidelines set down in
    Machado et al. (2018), "Revisiting the Arcade Learning Environment:
    Evaluation Protocols and Open Problems for General Agents".
    """

    def __init__(self,
                 environment,
                 frame_skip=4,
                 terminal_on_life_loss=False,
                 screen_size=84,
                 stack_size=4,
                 skip_init=1):
        """Constructor for an Atari 2600 preprocessor.

        Args:
          environment: Gym environment whose observations are preprocessed.
          frame_skip: int, the frequency at which the agent experiences the game.
          stack_size: int, the size of the stack that environment returns
          buffer_size: int, the size of the buffer to sample transitions from
          sample_idxs: [int, ...], ids of the frames to sample from the buffer,
                        size of the list should equal the stack_size
          skip_init: int, the number of frames to skip before storing frames into the buffer
          terminal_on_life_loss: bool, If True, the step() method returns
            is_terminal=True whenever a life is lost. See Mnih et al. 2015.
          screen_size: int, size of a resized Atari 2600 frame.
          bounding_box: [int, ...] bounding box to crop image

        Raises:
          ValueError: if frame_skip or screen_size are not strictly positive.
        """

        super().__init__(environment=environment,
                         frame_skip=frame_skip,
                         terminal_on_life_loss=terminal_on_life_loss,
                         screen_size=screen_size)

        self.stack_size = stack_size
        self.skip_init = skip_init

        self.observation_buffer = np.empty((self.screen_size, self.screen_size, self.stack_size),
                                           dtype=np.uint8)

    def reset(self):

        # reset environment and skip first few frames
        super().reset()
        for i in range(self.skip_init):
            frame, _, _, info = super().step(self.action_space.sample())

        # Perform random actions to fill the buffer
        for i in range(self.stack_size):
            frame, _, _, info = super().step(self.action_space.sample())
            self.observation_buffer[..., i] = frame

        return self.observation_buffer.copy()

    def step(self, action):
        observation, accumulated_reward, is_terminal, info = super().step(action)
        self.observation_buffer = np.roll(self.observation_buffer, shift=-1, axis=-1)
        self.observation_buffer[..., -1] = observation

        return self.observation_buffer, accumulated_reward, is_terminal, info

    @property
    def observation_space(self):
        # Return the observation space adjusted to match the shape of the processed
        # observations.
        return Box(low=0, high=255, shape=(self.screen_size, self.screen_size, self.stack_size),
                   dtype=np.uint8)

    @property
    def unwrapped(self):
        return self.environment.unwrapped

if __name__ == "__main__":
    env = AtariFrameStack(gym.make('SpaceInvadersNoFrameskip-v4'),
                          frame_skip=4,
                          stack_size=4,
                          skip_init=50,
                          terminal_on_life_loss=False,
                          screen_size=84)

    stacked_state = env.reset()
    # plot_frame_stack(stacked_state)
    for i in range(10):
        stacked_state, reward, terminal, info = env.step(0)
        plot_stack(stacked_state)
    # plot_frame_stack(stacked_state)
    plot_stack(env.observation_buffer)
