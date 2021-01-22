import numpy as np

from collections import defaultdict

from policy_gradient.memory.sample_batch import SampleBatch

from policy_gradient.memory.sample_batch import discount_cumsum

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

NOISE = 0.7

class Reinforce:
    def __init__(self, config=None):
        config = config or {}
        self.n_actions = config['n_actions']
        self.obs_shape = config['obs_shape']
        self.is_continuous = config.get('is_continuous', False)
        self.memory = defaultdict(list)

        n_outputs = self.n_actions * 2 if self.is_continuous else self.n_actions
        output_act_f = 'linear' if self.is_continuous else 'softmax'
        self.model = self._create_model(self.obs_shape, n_outputs, output_act_f)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=.01)
        ...

    def compute_action(self, obs):
        if self.is_continuous:
            return self._compute_action_continuous(obs)
        else:
            return self._compute_action_discrete(obs)

    def _compute_action_continuous(self, obs):
        p = self.model.predict(obs[None, ...])

        mean = p[0, :self.n_actions]
        log_std = p[0, self.n_actions:]
        std = np.exp(log_std)
        action = np.random.normal(mean, scale=std)
        action = np.clip(action, -1, 1)
        if np.isnan(action[0]):
            pass
        return action, p

    def _compute_action_discrete(self, obs):
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
        action_prob_old = sample_batch[SampleBatch.ACTION_PROB]

        return_ = discount_cumsum(rewards, gamma=0.99)

        norm_return_ = ((return_ - return_.mean()) / (return_.std() + 1e-10)).flatten()
        # actions = actions.flatten()
        # actions_matrix = np.zeros((actions.size, actions.max() + 1))
        # actions_matrix[np.arange(actions.size), actions] = 1
        #
        # with tf.GradientTape() as tape:
        #     action_probs = self.model(obs)
        #     log_p = tf.math.log(tf.reduce_sum(action_probs * actions_matrix, axis=-1))
        #     loss = tf.reduce_sum(-log_p * norm_return_)
            # p = tf.reduce_sum(action_probs * actions_matrix, axis=-1)
            # loss = -tf.reduce_sum(p * return_.flatten())

        with tf.GradientTape() as tape:
            action_probs = self.model(obs)
            means, log_std = tf.split(action_probs, 2, axis=-1, num=None, name='split')
            std = tf.exp(log_std)
            var = tf.square(std)
            pi = 3.1415926
            denom = (2 * pi * var) ** 0.5
            prob_num = tf.exp(-tf.square(actions - means) / (2 * var))
            prob = tf.divide(prob_num, denom)
            log_prob = tf.math.log(prob + 1e-10)
            loss = -tf.reduce_mean(norm_return_[..., None] * log_prob)
        print(loss)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        if 40.0:
            gradients, global_norm = tf.clip_by_global_norm(
                gradients, 40.0
            )
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def hot_encode_action(self, action):
        '''encoding the actions into a binary list'''

        action_encoded = np.zeros(self.n_actions, np.float32)
        action_encoded[action] = 1
        return action_encoded

    def _create_model(self, obs_shape, n_outputs, output_act_f='softmax'):

        ''' builds the model using keras'''
        model = Sequential()

        # input shape is of observations
        model.add(Dense(16, input_shape=obs_shape, activation="tanh"))
        # add a relu layer
        model.add(Dense(16, activation="tanh"))

        # output shape is according to the number of action
        # The softmax function outputs a probability distribution over the actions
        model.add(Dense(n_outputs, activation=output_act_f))
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
n_actions = env.action_space.n


from policy_gradient.cartpole_continuous import ContinuousCartPoleEnv
env = ContinuousCartPoleEnv()
n_actions = env.action_space.shape[0]
is_continuous = True

agent = Reinforce(dict(n_actions=n_actions,
                       obs_shape=env.observation_space.shape,
                       is_continuous=is_continuous))
import matplotlib.pyplot as plt
scores = []
for i in range(500):
    score = run_episode(env, agent)
    agent.update()
    print(i, score)
    scores.append(score)
plt.plot(scores)
plt.show()