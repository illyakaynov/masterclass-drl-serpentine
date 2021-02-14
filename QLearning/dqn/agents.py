import math
import random

import numpy as np

from QLearning.dqn import ReplayBuffer

from os.path import join


class Agent:
    """
    Interface for the agent class
    """

    def __init__(self):
        self.stats = {}

    def compute_action(self, obs):
        ...

    def update(self, *args, **kwargs):
        ...

    def finalize_episode(self, *args, **kwargs):
        ...


class RandomAgent(Agent):
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.epsilon = 1.0

    def compute_action(self, obs):
        return random.randrange(self.n_actions)


class EpsilonGreedyAgent(Agent):
    def __init__(
        self,
        n_actions,
        network=None,
        gamma=0.99,
        batch_size=32,
        replay_capacity=10000,
        training_start=50,  # start training after x number of steps
        training_interval=4,  # train every x steps
        start_epsilon=1.0,
        end_epsilon=0.02,
        epsilon_decay=5e-6,
        root_folder="saved_run",
        save_best=False,
        save_interval=None,
        uint_as_obs=False,  # backwards compatibility with old pre-trained networks

    ):

        self.n_actions = n_actions
        self.network = network

        self.replay_buffer = ReplayBuffer(replay_capacity)

        self.epsilon = start_epsilon
        self.max_epsilon = start_epsilon
        self.min_espilon = end_epsilon
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.gamma = gamma

        self.steps = 0
        self.runs = 0
        self.return_ = 0

        self.training_start = training_start
        self.training_interval = training_interval

        self.episode_losses = []

        self.best_return = -100000
        self.root_folder = root_folder
        self.save_best = save_best
        self.save_interval = save_interval

        # In the previous iteration of the algorithm The input had uint type.
        # Setting this in case we would like to load old networks
        self.uint_as_obs = uint_as_obs

    def compute_action(self, obs):
        """
        Given the state return the action following epsilon-greedy strategy
        :param
            state (array_like): state of the system
        :return(int): action
        """
        if self.network is None:
            message = "The network is not set. Please, create with Agent(network=network) or use agent.load()"
            print(message)
            raise ValueError()

        # When state is missing the batch dimension add it
        if len(obs.shape) == 3 or len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)

        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            if obs.dtype == "uint8" and not self.uint_as_obs:
                obs = obs.astype("float32") / 255.0

            action = self.network.get_best_action(obs)
            # Since this get_best_action() returns a tensor
            # we convert it to numpy and then to the Python int
            action = action.numpy().item()

        return action

    def update(self, current_state, action, reward, next_state, is_terminal):
        """
        Append the transition to a replay buffer and train the network if necessary
        :param current_state: (array_like)
        :param action: (int)
        :param reward: (float)
        :param next_state: (array_like)
        :param is_terminal: (float)
        :return: loss of the network
        """
        self.return_ = reward + self.gamma * self.return_
        self.replay_buffer.store(
            (
                current_state,
                action,
                np.asarray(reward, dtype="float32"),
                next_state,
                np.asarray(is_terminal, dtype="float32"),
            )
        )

        loss = 0
        train_network = (
            self.training_start < self.steps
            and self.steps % self.training_interval == 0
        )
        if train_network:
            sample_batch = self.replay_buffer.sample_batch(self.batch_size)
            loss = self.network.update(*sample_batch)
            self.episode_losses.append(loss)
        self.steps += 1

        return loss

    def finalize_episode(self):
        """
        Clear score, return episode statistics values. Update epsilon. Save the model if necessary.
        :return: agents statistics per episode
        """
        stats = {}
        # Save the best model so far
        if self.save_best and self.return_ > self.best_return and self.runs > 200:
            self.best_return = self.return_
            self.save_model(join(self.root_folder, "best"))

        if self.save_interval and self.runs % self.save_interval == 0:
            self.save_model(join(self.root_folder, f"run{self.runs}"))

        stats["runs"] = self.runs
        self.runs += 1

        # reset episode return
        stats["return"] = self.return_
        self.return_ = 0

        # update exploration
        stats["epsilon"] = self.epsilon
        self.update_epsilon(self.steps)

        # get mean of the losses for this episode
        loss = np.mean(self.episode_losses) if self.episode_losses else 0
        stats["loss"] = loss
        self.episode_losses = []

        stats["steps"] = self.steps

        return stats

    def update_epsilon(self, time_step):
        """
        Update epsilon value based on the time-step
        :param time_step: current timestep
        :return:
        """
        self.epsilon = self.min_espilon + (
            self.max_epsilon - self.min_espilon
        ) * math.exp(-self.epsilon_decay * time_step)

    def save_model(self, filepath="saved_model"):
        self.network.save(filepath=filepath)

    def load_model(self, filepath="saved_model"):
        self.network.load(filepath=filepath)
