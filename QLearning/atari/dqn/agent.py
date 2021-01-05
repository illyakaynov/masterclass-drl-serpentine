import numpy as np

from dqn.q_network import DQN
from dqn.exploration import EpsilonGreedyDecay

from dqn.replay_memory.replay_memory import CircularBufferReplayMemory, PrioritisedCircularBufferReplayMemory


class QAgent:

    def __init__(self,
                 observation_shape: tuple,
                 n_actions: int,
                 gamma=0.99,
                 training_start=100,
                 training_interval=4,
                 batch_size=32,
                 priority_replay=False,
                 replay_buffer_params=None,
                 network_params=None,
                 exploration_params=None, ):
        self.gamma = gamma
        self.training_start = training_start  # start training after X game frames
        self.training_interval = training_interval  # run a training step every X game frames
        self.batch_size = batch_size  # batch size for training NN
        self.n_actions = n_actions
        self.observation_shape = observation_shape
        self.batch_size = batch_size
        self.priority_replay = priority_replay

        if network_params is None:
            network_params = {}
        self.q_network = DQN(input_shape=observation_shape,
                             n_outputs=n_actions,
                             **network_params)

        if replay_buffer_params is None:
            replay_buffer_params = {}

        if not self.priority_replay:
            self.replay_memory = CircularBufferReplayMemory(self.observation_shape,
                                                            batch_size=batch_size,
                                                            **replay_buffer_params)
        else:
            self.replay_memory = PrioritisedCircularBufferReplayMemory(self.observation_shape,
                                                            batch_size=batch_size,
                                                            **replay_buffer_params)

        self.exploration = EpsilonGreedyDecay(**exploration_params)

        self.steps = 0
        self.return_ = 0

    def get_action(self, x) -> int:
        if self.exploration.do_exploration():
            action = np.random.randint(self.n_actions)
        else:
            action = self.q_network.get_action_online(np.expand_dims(x, axis=0)).numpy()
            action = int(action)
        return action

    def update(self, state, action, reward, next_state, is_terminal):
        self.return_ = reward + self.gamma * self.return_
        self.replay_memory.insert((
            state,
            action,
            reward,
            next_state,
            is_terminal))

        if self.training_start < self.replay_memory.add_count:
            if self.steps % self.training_interval == 0 and self.steps != 0:
                sample_batch = self.replay_memory.sample_memories()
                self.q_network.train_online(sample_batch)

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
