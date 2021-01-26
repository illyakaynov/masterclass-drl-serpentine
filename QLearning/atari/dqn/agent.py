import numpy as np

from dqn.q_network import DQN
from dqn.exploration import EpsilonGreedyDecay

from dqn.replay_memory.replay_memory import (
    CircularBufferReplayMemory,
    PrioritisedCircularBufferReplayMemory,
)


class QAgent:
    def __init__(
        self,
        observation_shape: tuple,
        n_actions: int,
        gamma=0.99,
        batch_size=32,
        training_interval=4,
        training_start=100,
        priority_replay=False,
        replay_buffer_config=None,
        network_config=None,
        exploration_config=None,
    ):
        self.gamma = gamma
        self.training_start = training_start  # start training after X game frames
        self.training_interval = (
            training_interval  # run a training step every X game frames
        )
        self.batch_size = batch_size  # batch size for training NN
        self.n_actions = n_actions
        self.observation_shape = observation_shape
        self.batch_size = batch_size
        self.priority_replay = priority_replay

        network_config = network_config or {}
        self.q_network = DQN(
            input_shape=observation_shape, n_outputs=n_actions, gamma=gamma, **network_config
        )

        replay_buffer_config = replay_buffer_config or {}
        if not self.priority_replay:
            self.replay_memory = CircularBufferReplayMemory(
                self.observation_shape,
                batch_size=batch_size,
                observation_dtype=np.uint8 if len(self.observation_shape) > 1 else np.float32,
                **replay_buffer_config
            )
        else:
            self.replay_memory = PrioritisedCircularBufferReplayMemory(
                self.observation_shape, batch_size=batch_size, **replay_buffer_config
            )

        self.exploration = EpsilonGreedyDecay(**exploration_config)

        self.steps = 0
        self.return_ = 0

    def get_action(self, x) -> int:
        if self.exploration.do_exploration():
            action = np.random.randint(self.n_actions)
        else:
            action = self.q_network.get_action_online(np.expand_dims(x, axis=0)).numpy()
            action = int(action)
        return action

    def save_experience(self, state, action, reward, next_state, is_terminal):
        self.return_ = reward + self.gamma * self.return_
        self.replay_memory.insert((state, action, reward, next_state, is_terminal))

        if self.training_start < self.replay_memory.add_count:
            if self.steps % self.training_interval == 0 and self.steps != 0:
                sample_batch = self.replay_memory.sample_memories()
                loss, td_error, indices = self.q_network.train_online(sample_batch)

                if self.priority_replay:
                    self.replay_memory.update_priorities(indices, td_error)

            if self.steps % self.q_network.copy_steps == 0:
                self.q_network.update()

        self.steps += 1

    def finalize_episode(self):
        # reset episode return
        self.return_ = 0
        # update exploration
        self.exploration.update()
        # get mean of the losses for this episodez
        loss = 0
        if self.q_network.loss_history:
            loss = np.mean(self.q_network.loss_history)
            self.q_network.loss_history = []
        return loss
