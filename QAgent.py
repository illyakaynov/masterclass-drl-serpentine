import numpy as np

# For plotting metrics
all_epochs = []
all_penalties = []


class QAgent:
    def __init__(
        self, n_states, n_actions, epsilon=0.2, alpha=0.1, gamma=0.9, learn=True
    ):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.n_actions = n_actions
        self.n_states = n_states
        self.learn = learn

        self.sum_rewards = 0

        self.reset()

    def reset(self):
        self.sum_rewards = 0
        self.q_table = np.zeros((self.n_states, self.n_actions), np.float)

    def compute_action(self, obs):
        if self.learn and (np.random.random() < self.epsilon):
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.q_table[obs, :])
        return action

    def update(self, obs, action, next_obs, reward, done):
        if self.learn:
            old_q = self.q_table[obs, action]
            td_update = (
                self.gamma * np.max(self.q_table[next_obs, :])
                - self.q_table[obs, action]
            )
            self.q_table[obs, action] = old_q + self.alpha * (reward + td_update)
            self.sum_rewards += reward

