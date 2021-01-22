import sys

import gym
import numpy as np
import tensorflow as tf
from memory.sample_batch import SampleBatch, discount_cumsum
from tensorboardX import SummaryWriter
from tensorflow.keras import layers, models, optimizers

import tensorflow.keras.backend as K

from collections import defaultdict


logdir = "test1"
writer = tf.summary.create_file_writer(logdir + "/metrics")
writer.set_as_default()

default_config = dict(
    explore=True,
    obs_shape=(8,),
    num_actions=2,
    continuous=False,
    clip_value=0.2,
    gamma=0.99,
    num_sgd_iter=6,
    num_epochs=20,
    sgd_minibatch_size=128,
    train_batch_size=4000,
    num_dim_critic=(64, 64),
    act_f_critic="tanh",
    num_dim_actor=(64, 64),
    act_f_actor="tanh",
    vf_share_layers=False,
    entropy_coeff=1e-5,
    lr_actor=0.01,
    lr_critic=0.001,
    clip_gradients_by_norm=None,
)


class PPOAgent:
    def __init__(self, config=None):
        config = config or {}
        self.config = config = {**default_config, **config}
        self.actor = build_actor_network(
            obs_shape=config["obs_shape"],
            num_actions=config["num_actions"],
            num_dim=config["num_dim_actor"],
            act_f=config["act_f_actor"],
        )

        self.critic = build_critic_network(
            obs_shape=config["obs_shape"],
            num_dim=config["num_dim_critic"],
            act_f=config["act_f_critic"],
        )

        self.num_actions = config["num_actions"]
        self.explore = config["explore"]
        self.continuous = config["continuous"]
        self.train_batch_size = config["train_batch_size"]
        self.sgd_minibatch_size = config["sgd_minibatch_size"]
        self.num_epochs = config["num_epochs"]
        self.num_sgd_iter = config["num_sgd_iter"]

        self.gamma = config["gamma"]
        self.lr_actor = config["lr_actor"]
        self.lr_critic = config["lr_critic"]
        self.clip_value = config["clip_value"]
        self.entropy_coeff = config["entropy_coeff"]

        self.sgd_iters = 0
        self.total_epochs = 0

        self.clip_gradients_by_norm = config["clip_gradients_by_norm"]

        self.actor_optimizer = optimizers.Adam(self.lr_actor)
        self.critic_optimizer = optimizers.Adam(self.lr_critic)

    def _compute_action_discrete(self, obs):
        p = self.actor.predict(obs[None, :])
        if self.explore:
            action = np.random.choice(self.num_actions, p=np.nan_to_num(p[0]))
        else:
            action = np.argmax(p[0])
        action_matrix = np.zeros(self.num_actions, dtype=np.float32)
        action_matrix[action] = 1.0
        return action, action_matrix, p

    def get_action_continuous(self, obs):
        p = self.actor.predict(obs[None, ...])
        action = action_matrix = p[0] + np.random.normal(loc=0, scale=1.0, size=p[0].shape)

        return action, action_matrix, p

    def compute_action(self, obs):
        if self.continuous:
            ...
        else:
            action, action_matrix, p = self._compute_action_discrete(obs)
        # print(p)
        return action, action_matrix, p

    def get_batch(self):
        observations = []
        next_observations = []
        actions = []
        rewards = []
        dones = []
        infos = []
        probs = []
        eps_ids = []

        episode_id = 0
        score = 0
        episode_rewards = []
        obs = env.reset()
        for __ in range(self.train_batch_size):
            # obs = self.preprocess_obs(obs)
            action, action_matrix, prob = self.compute_action(obs)
            next_obs, reward, done, info = env.step(action)
            score += reward
            observations.append(obs)
            actions.append(action_matrix)
            dones.append(done)
            next_observations.append(next_obs)
            infos.append(info)
            probs.append(prob)
            eps_ids.append(episode_id)
            episode_rewards.append(reward)
            obs = next_obs
            if done:
                # episode_rewards[-1] = 0
                # print(score)
                score = 0
                obs = env.reset()
                episode_id += 1
                return_ = discount_cumsum(
                    np.asarray(episode_rewards), self.gamma
                )
                return_ = ((return_ - return_.mean()) / (return_.std() + 1e-10))
                rewards += return_.tolist()
                episode_rewards = []
        return_ = discount_cumsum(np.asarray(episode_rewards), self.gamma)
        return_ = ((return_ - return_.mean()) / (return_.std() + 1e-10))
        rewards += return_.tolist()
        return SampleBatch(
            {
                SampleBatch.OBS: observations,
                SampleBatch.ACTIONS: actions,
                SampleBatch.DONES: dones,
                SampleBatch.INFOS: infos,
                SampleBatch.REWARDS: rewards,
                SampleBatch.ACTION_PROB: np.reshape(probs, (-1, self.num_actions)),
                SampleBatch.EPS_ID: eps_ids,
            }
        )

    def run(self):

        history = defaultdict(list)

        while self.sgd_iters < self.num_sgd_iter:
            epoch_actor_loss = []
            epoch_critic_loss = []

            train_batch = self.get_batch()
            obs = train_batch[SampleBatch.OBS]
            action = train_batch[SampleBatch.ACTIONS]
            action_prob = train_batch[SampleBatch.ACTION_PROB]
            values = train_batch[SampleBatch.REWARDS].astype("float32")

            # pred_values = 0

            dataset = tf.data.Dataset.from_tensor_slices(
                (obs, values, action_prob, action)
            ).batch(self.sgd_minibatch_size).shuffle(1)
            for obs_batch, values_batch, action_prob_batch, action_batch in dataset:

                with tf.GradientTape() as tape:
                    pred_values = self.critic(obs_batch)
                    advantage_batch = values_batch - pred_values
                    critic_loss = K.mean(K.square(advantage_batch)) * 0.01

                critic_gradients = tape.gradient(
                    critic_loss, self.critic.trainable_variables
                )
                if self.clip_gradients_by_norm:
                    critic_gradients, global_norm = tf.clip_by_global_norm(
                        critic_gradients, self.clip_gradients_by_norm
                    )
                self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
                epoch_critic_loss.append(critic_loss)

                with tf.GradientTape() as tape:
                    y_pred = self.actor(obs_batch)
                    y_true = action_batch

                    prob = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
                    old_prob = tf.reduce_sum(y_true * action_prob_batch, axis=-1, keepdims=True)
                    prob_ratio = prob / (old_prob + 1e-10)
                    surrogate = prob_ratio * advantage_batch
                    surrogate_cliped = (
                        K.clip(prob_ratio, 1 - self.clip_value, 1 + self.clip_value)
                        * advantage_batch
                    )
                    actor_loss = -tf.reduce_mean(
                        tf.minimum(surrogate, surrogate_cliped)
                        # + self.entropy_coeff * -(prob * tf.math.log(prob + 1e-10))
                    )
                    entropy = -(prob * tf.math.log(prob + 1e-10))
                    epoch_actor_loss.append(actor_loss.numpy())

                actor_gradients = tape.gradient(
                    actor_loss, self.actor.trainable_variables
                )
                if self.clip_gradients_by_norm:
                    actor_gradients, global_norm = tf.clip_by_global_norm(
                        actor_gradients, self.clip_gradients_by_norm
                    )
                self.actor_optimizer.apply_gradients(
                    zip(actor_gradients, self.actor.trainable_variables)
                )

                for i in range(len(actor_gradients)):
                    tf.summary.histogram(
                        f"actor_gradient_{i}",
                        actor_gradients[i],
                        step=self.total_epochs,
                    )
                for i in range(len(self.actor.trainable_variables)):
                    tf.summary.histogram(
                        f"actor_weights_{i}",
                        self.actor.trainable_variables[i],
                        step=self.total_epochs,
                    )
                tf.summary.scalar('entropy', tf.reduce_mean(entropy), step=self.total_epochs)
                for i in range(len(prob)):
                    tf.summary.scalar(f'action_prob_{i}', tf.reduce_mean(prob[i]), step=self.total_epochs)

                self.total_epochs += 1

            history["actor_loss"].append(np.mean(epoch_actor_loss))
            history["critic_loss"].append(np.mean(epoch_critic_loss))

            tf.summary.scalar("Actor loss", history["actor_loss"][-1], self.sgd_iters)
            tf.summary.scalar("Critic loss", history["critic_loss"][-1], self.sgd_iters)
            print(
                self.sgd_iters,
                history["actor_loss"][-1],
                history["critic_loss"][-1],
            )
            self.sgd_iters += 1
            val_score = np.mean([self.run_episode() for i in range(1)])
            history["score"].append(val_score)
            print(val_score)

        return history

    def run_episode(self):
        done = False
        score = 0
        obs = env.reset()
        while not done:
            action, __, __ = self.compute_action(obs)
            obs, reward, done, info = env.step(action)
            score += reward
        return score


def build_critic_network(
    obs_shape, num_dim=(64, 64), act_f="tanh"
):
    state_input = layers.Input(shape=obs_shape, dtype=tf.float32)
    x = state_input
    for i, dim in enumerate(num_dim):
        x = layers.Dense(dim, activation=act_f, name=f"hidden_{i}")(x)

    out_value = layers.Dense(1, name="output")(x)

    model = models.Model(inputs=state_input, outputs=out_value, name="critic")
    model.summary()
    return model


def build_actor_network(
    obs_shape, num_actions, num_dim=(64, 64), act_f="tanh", output_act_f='softmax'
):
    state_input = layers.Input(shape=obs_shape, dtype=tf.float32)

    x = state_input
    for i, dim in enumerate(num_dim):
        x = layers.Dense(dim, activation=act_f, name=f"hidden_{i}")(x)

    out_actions = layers.Dense(num_actions, activation=output_act_f, name="output")(x)

    model = models.Model(inputs=state_input, outputs=out_actions, name="actor")
    model.summary()
    return model


def ppo_continuous_loss(advantage, old_prediction, noise=1.0, clip_value=0.2):
    def loss(y_true, y_pred):
        var = K.square(noise)
        pi = 3.1415926
        denom = K.sqrt(2 * pi * var)
        prob_num = K.exp(-K.square(y_true - y_pred) / (2 * var))
        old_prob_num = K.exp(-K.square(y_true - old_prediction) / (2 * var))

        prob = prob_num / denom
        old_prob = old_prob_num / denom
        r = prob / (old_prob + 1e-10)

        return -K.mean(
            K.minimum(
                r * advantage,
                K.clip(r, min_value=1 - clip_value, max_value=1 + noise) * advantage,
            )
        )

    return loss


if __name__ == "__main__":

    from gym.wrappers import TransformObservation

    # tf.keras.backend.set_floatx("float64")
    # env = gym.make("LunarLander-v2")
    env = TransformObservation(
        gym.make("CartPole-v1")
        # gym.make("LunarLander-v2")
        , f=lambda x: x.astype(np.float32)
    )

    agent = PPOAgent(
        config=dict(
            obs_shape=env.observation_space.shape,
            num_actions=env.action_space.n,
            num_dim_critic=(16, 16),
            act_f_critic="tanh",
            num_dim_actor=(16, 16),
            act_f_actor="tanh",
            num_sgd_iter=100,
            num_epochs=10,
            sgd_minibatch_size=32,
            train_batch_size=200,
            clip_gradients_by_norm=40.0,
        )
    )
    history = agent.run()
    from matplotlib import pyplot as plt

    plt.plot(history["score"])
    plt.plot(history["actor_loss"])
    plt.plot(history["critic_loss"])
    plt.show()
