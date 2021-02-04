import os
from collections import defaultdict
from IPython.core import display

import matplotlib.pyplot as plt

import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from gym.spaces import Box, Discrete
from gym.wrappers import Monitor
from policy_gradient.memory.sample_batch import (
    SampleBatch,
    compute_advantages,
    discount_cumsum,
    standardized,
)
from policy_gradient.cartpole_continuous import ClipActionsWrapper, ContinuousCartPoleEnv
from policy_gradient.networks import build_actor_network, build_critic_network
from policy_gradient.utis import (
    compute_entropy_discrete,
    compute_entropy_gaussian,
    compute_log_p_discrete,
    compute_log_p_gaussian,
    one_hot_encode,
)
from tensorflow.keras import optimizers

import seaborn as sns

default_config = dict(
    logdir="default",
    explore=True,
    gamma=0.99,
    num_epochs=20,
    train_batch_size=4000,
    num_dim_actor=(64, 64),
    act_f_actor="tanh",
    entropy_coeff=1e-5,
    lr=0.01,
    clip_gradients_by_norm=None,
    num_eval_episodes=1,
    standardize_return=True,
)

import yaml


class ReinforceAgent:
    def __init__(self, config=None):
        config = config or {}

        self.config = config = {**default_config, **config}

        self.logdir = config["logdir"]
        os.makedirs(self.logdir, exist_ok=True)
        yaml.dump(config, open(os.path.join(self.logdir, "config.yaml"), "w"))

        writer = tf.summary.create_file_writer(self.logdir)
        writer.set_as_default()

        self.env = config["env_or_env_name"]
        if isinstance(self.env, str):
            self.env = gym.make(self.env)

        self.continuous = True if isinstance(self.env.action_space, Box) else False

        if self.continuous:
            self.num_outputs = self.env.action_space.shape[0] * 2
            self.env = ClipActionsWrapper(self.env)
        else:
            self.num_outputs = self.env.action_space.n

        self.explore = config["explore"]
        self.num_epochs = config["num_epochs"]

        self.gamma = config["gamma"]
        self.lr = config["lr"]
        self.entropy_coeff = config["entropy_coeff"]

        self.sgd_iters = 0
        self.total_epochs = 0

        self.clip_gradients_by_norm = config["clip_gradients_by_norm"]
        self.standardize_advantages = config["standardize_return"]

        self.num_eval_episodes = config["num_eval_episodes"]

        self.network = build_actor_network(
            obs_shape=self.env.observation_space.shape,
            n_outputs=self.num_outputs,
            num_dim=config["num_dim_actor"],
            act_f=config["act_f_actor"],
            output_act_f="linear" if self.continuous else "softmax",
        )

        self.actor_optimizer = optimizers.Adam(self.lr, epsilon=1e-5)

    def _compute_action_discrete(self, obs):
        action_probs = self.network.predict(obs[None, :])
        if self.explore:
            action = np.random.choice(
                self.num_actions, p=np.nan_to_num(action_probs[0])
            )
        else:
            action = np.argmax(action_probs[0])
        return action, action_probs

    def _compute_action_continuous(self, obs):
        means_and_log_stds = self.network.predict(obs[None, ...])
        means, log_stds = np.split(means_and_log_stds.squeeze(), 2)
        stds = np.exp(log_stds)
        stds = stds if self.explore else np.zeros_like(log_stds)
        action = np.random.normal(loc=means, scale=stds)
        return action, means_and_log_stds

    def compute_action(self, obs):
        if self.continuous:
            action, action_probs = self._compute_action_continuous(obs)
        else:
            action, action_probs = self._compute_action_discrete(obs)

        return action, action_probs

    def sample_trajectory(self):
        traj_dict = defaultdict(list)

        obs = self.env.reset()
        done = False
        score = 0
        while not done:
            action, action_prob = self.compute_action(obs)
            next_obs, reward, done, info = self.env.step(action)

            traj_dict[SampleBatch.OBS].append(obs)
            traj_dict[SampleBatch.ACTIONS].append(action)
            traj_dict[SampleBatch.DONES].append(done)
            traj_dict[SampleBatch.REWARDS].append(reward)
            score += reward
            if self.continuous:
                traj_dict[SampleBatch.MEANS_AND_LOG_STDS].append(action_prob)
            else:
                traj_dict[SampleBatch.ACTION_PROB].append(action_prob)

            obs = next_obs
        sample_batch = SampleBatch(traj_dict)
        sample_batch[SampleBatch.RETURNS] = discount_cumsum(sample_batch[SampleBatch.REWARDS], self.gamma)
        return sample_batch, score

    # @tf.function
    def train_op(self, obs, actions_old, returns):
        with tf.GradientTape() as tape:
            if self.continuous:
                means_and_log_stds = self.network(obs)
                log_p = compute_log_p_gaussian(
                    actions_old, means_and_log_stds
                )
            else:
                action_prob = self.network(obs)
                log_p = compute_log_p_discrete(actions_old, action_prob)

            surrogate = log_p * returns

            # entropy = - sum_x x * log(x)
            if self.continuous:
                entropy_bonus = compute_entropy_gaussian(means_and_log_stds)
            else:
                entropy_bonus = compute_entropy_discrete(action_prob)
            entropy_bonus = tf.reduce_mean(entropy_bonus)
            loss = -(
                    tf.reduce_mean(surrogate)
                    + self.entropy_coeff * entropy_bonus
            )

            actor_gradients = tape.gradient(
                loss, self.network.trainable_variables
            )
            if self.clip_gradients_by_norm:
                actor_gradients, global_norm = tf.clip_by_global_norm(
                    actor_gradients, self.clip_gradients_by_norm
                )
            self.actor_optimizer.apply_gradients(
                zip(actor_gradients, self.network.trainable_variables)
            )
            return loss, entropy_bonus

    def run(self, plot_stats=None, plot_period=1):

        if plot_stats:
            num_plots = len(plot_stats)
            fig, axs = plt.subplots(num_plots, 1, squeeze=False, figsize=(10, 5 * num_plots))
            axs = axs.ravel()

        history = defaultdict(list)

        for i in range(self.num_epochs):
            stats = {}
            epoch_actor_loss = []
            epoch_critic_loss = []

            train_batch, score = self.sample_trajectory()

            obs = train_batch[SampleBatch.OBS]
            actions_old = train_batch[SampleBatch.ACTIONS]

            if not self.continuous:
                actions_old = one_hot_encode(actions_old, self.num_outputs)

            returns = train_batch[SampleBatch.RETURNS].astype("float32")
            returns = standardized(returns.squeeze())
            actor_loss, entropy_bonus = self.train_op(obs, actions_old, returns)

            epoch_actor_loss.append(actor_loss.numpy())

            tf.summary.scalar(
                "entropy", entropy_bonus, step=self.total_epochs
            )

            self.total_epochs += 1

            stats["actor_loss"] = np.mean(epoch_actor_loss)
            tf.summary.scalar("Actor loss", actor_loss, self.sgd_iters)
            self.sgd_iters += 1

            stats["score"] = score

            if plot_stats:
                if i % plot_period == 0:
                    for ax, stat_name in zip(axs, plot_stats):
                        ax.clear()
                        # print(stat_name, len(history[stat_name]))

                        sns.lineplot(
                            x=np.arange(len(history[stat_name])), y=history[stat_name], ax=ax
                        )

                        ax.set_title(stat_name)
                    display.display(fig)
                    display.clear_output(wait=True)

            else:
                print(f"episode {i}/{self.num_epochs} | {stats}", )

            for k, v in stats.items():
                history[k].append(v)

        return history


def run_episode(env, agent, monitor=True, logdir=None):
    done = False
    score = 0

    if monitor:
        env = Monitor(
            env,
            logdir,
            video_callable=lambda x: True,
            force=True,
        )

    obs = env.reset()
    while not done:
        action, __ = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        score += reward
    env.close()
    return score


if __name__ == "__main__":
    agent = ReinforceAgent(
        config=dict(
            env_or_env_name=ContinuousCartPoleEnv(),
            logdir=os.path.join("cartpole_continuous", "first_try"),
            explore=True,
            gamma=0.99,
            num_epochs=200,
            num_dim_actor=[32, 32],
            act_f_actor="relu",
            entropy_coeff=1e-3,
            lr=0.025,
            clip_gradients_by_norm=None,
        )
    )

    history = agent.run()
    from matplotlib import pyplot as plt

    plt.plot(history["score"])
    plt.plot(history["actor_loss"])
    plt.plot(history["critic_loss"])
    plt.show()

    # from time import time
    #
    # start = time()
    # training_batch = sample_batch(env, agent, 200)
    # print(time() - start)
