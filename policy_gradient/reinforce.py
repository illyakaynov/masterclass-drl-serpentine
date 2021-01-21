import numpy as np

from collections import defaultdict

from policy_gradient.memory.sample_batch import SampleBatch

from policy_gradient.memory.sample_batch import discount_cumsum

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

class Reinforce:
    def __init__(self, config=None):
        config = config or {}
        self.n_actions = config['n_actions']
        self.obs_shape = config['obs_shape']
        self.is_continuous = config.get('is_continuous', False)
        self.memory = defaultdict(list)
        self.model = self._create_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        ...

    def compute_action(self, obs):
        action_prob = self.model.predict(obs[None, ...])
        action_prob = action_prob.flatten()
        action = np.random.choice(self.n_actions, p=np.nan_to_num(action_prob))

        return action, action_prob

    def save_experience(self, exp):
        for k, v in exp.items():
            self.memory[k].append(v)

    def update(self):
        sample_batch = SampleBatch(self.memory)
        self.memory = defaultdict(list)

        obs = sample_batch[SampleBatch.OBS]
        rewards = sample_batch[SampleBatch.REWARDS]
        actions = sample_batch[SampleBatch.ACTIONS]
        action_prob = sample_batch[SampleBatch.ACTION_PROB]

        return_ = discount_cumsum(rewards, gamma=0.99)

        norm_return_ = ((return_ - return_.mean()) / (return_.std() + 1e-10)).flatten()
        actions = actions.flatten()
        actions_matrix = np.zeros((actions.size, actions.max() + 1))
        actions_matrix[np.arange(actions.size), actions] = 1

        with tf.GradientTape() as tape:
            action_probs = self.model(obs)
            log_p = tf.math.log(tf.reduce_sum(action_probs * actions_matrix, axis=-1))
            loss = tf.reduce_sum(-log_p * norm_return_)
            # p = tf.reduce_sum(action_probs * actions_matrix, axis=-1)
            # loss = -tf.reduce_sum(p * return_.flatten())

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def hot_encode_action(self, action):
        '''encoding the actions into a binary list'''

        action_encoded = np.zeros(self.n_actions, np.float32)
        action_encoded[action] = 1
        return action_encoded

    def _create_model(self):

        ''' builds the model using keras'''
        model = Sequential()

        # input shape is of observations
        model.add(Dense(16, input_shape=self.obs_shape, activation="tanh"))
        # add a relu layer
        model.add(Dense(16, activation="tanh"))

        # output shape is according to the number of action
        # The softmax function outputs a probability distribution over the actions
        model.add(Dense(self.n_actions, activation="softmax"))
        # model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01))
        return model

def run_episode(env, agent):
    done = False
    score = 0
    obs = env.reset()
    while not done:
        action, action_prob = agent.compute_action(obs)
        next_obs, reward, done, info = env.step(action)
        agent.save_experience(
            {
                SampleBatch.OBS: obs,
                SampleBatch.REWARDS: reward,
                SampleBatch.ACTIONS: action,
                SampleBatch.ACTION_PROB: action_prob,
                SampleBatch.NEXT_OBS: next_obs,
                SampleBatch.DONES: done,
                SampleBatch.INFOS: info,
            }
        )
        obs = next_obs
        score += reward
    return score


import gym

env = gym.make('CartPole-v0')
agent = Reinforce(dict(n_actions=env.action_space.n,
                       obs_shape=env.observation_space.shape))
import matplotlib.pyplot as plt
scores = []
for i in range(20):
    score = run_episode(env, agent)
    agent.update()
    print(i, score)
    scores.append(score)
plt.plot(scores)
plt.show()