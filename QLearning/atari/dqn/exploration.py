
import numpy as np

class EpsilonGreedyDecay:
    def __init__(self,
                 epsilon_init=1.0,
                 epsilon_min=0.02,
                 epsilon_decay=0.95):
        super().__init__()
        self.epsilon_init = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon = self.epsilon_init

    def do_exploration(self):
        if np.random.random() < self.epsilon:  # explore
            return True
        return False

    def update(self):
        self.epsilon = max(self.epsilon_decay * self.epsilon, self.epsilon_min)
