import random

from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, replay_capacity=50000, include_last_samples=3):
        """
        Simple replay buffer based on the collections.deque
        :param replay_capacity: the size of the buffer, i.e. the number of last transitions to save
        :param include_last_samples: include a number of most recent observations into a sample batch
        """
        self.replay_capacity = replay_capacity

        self.include_last = include_last_samples

        self._buffer = deque(maxlen=self.replay_capacity)

    def store(self, transition):
        """
        store the transition in a replay buffer
        :param transition:
        :return: None
        """
        self._buffer.appendleft(transition)

    def sample_batch(self, batch_size):
        """
        Sample a batch of transitions from replay buffer
        :param batch_size: size of the sampled batch
        :return: tuple of ndarrays with batch_size as first dimension
        """
        batch = random.sample(self._buffer, batch_size)
        for i in range(self.include_last):
            batch[i] = self._buffer[i]
        return self._batch_to_arrays(batch)

    def _batch_to_arrays(self, batch):
        """
        Transforms list of transition tuples to a tuple of ndarrays
        :param batch: list of tuples of every element in a batch
        :return: tuple of ndarrays
        """
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        terminals = np.array([x[4] for x in batch])

        return states, actions, rewards, next_states, terminals

    @property
    def buffer_size(self):
        """
        :return: Current size of the buffer
        """
        return len(self._buffer)
