import os
from collections import defaultdict
from os.path import join

import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import yaml
from gym.spaces import Box
from IPython.core import display
from policy_gradient.action_dist import CategoricalDistribution, GaussianDistribution
from policy_gradient.cartpole_continuous import (
    ClipActionsWrapper,
    ContinuousCartPoleEnv,
)
from policy_gradient.memory.sample_batch import (
    SampleBatch,
    discount_cumsum,
    np_standardized,
)
from policy_gradient.networks import build_actor_network
from tensorflow.keras import optimizers

DEFAULT_CONFIG = dict(
    # Folder where to save files related to the run
    logdir=join("Experiments", "pg_default"),
    # True to use tensorboard for logging
    use_tensorboard=False,
    # True to perform non-deterministic during an episode
    explore=True,
    # Discount for rewards
    gamma=0.99,
    # Number of episodes to run the algorithm
    num_episodes=20,
    # Dimensions of the dense layers of the network
    num_dim_actor=(64, 64),
    # Activation function in dense layers
    act_f_actor="tanh",
    # Entropy coefficient, used to control exploration
    entropy_coeff=1e-5,
    # Learning rate for the neural network optimizer
    lr=0.01,
    # If specified, clip the global norm of gradients by this amount.
    clip_gradients_by_norm=None,
    # True to normalize the returns (0 mean, 1 variance) on the batch level
    standardize_return=True,
)


class ReinforceAgent:
    def __init__(self, config=None):

        config = config or {}
        self.config = config = {**DEFAULT_CONFIG, **config}

        # create environment
        self.env = config["env_or_env_name"]
        if isinstance(self.env, str):
            self.env = gym.make(self.env)

        # set up distributions and action clip wrappers
        self.continuous = True if isinstance(self.env.action_space, Box) else False
        if self.continuous:
            self.num_outputs = self.env.action_space.shape[0] * 2
            self.env = ClipActionsWrapper(self.env)
            self.action_dist_cls = GaussianDistribution
        else:
            self.num_outputs = self.env.action_space.n
            self.action_dist_cls = CategoricalDistribution

        self.explore = config["explore"]
        self.gamma = config["gamma"]
        self.entropy_coeff = config["entropy_coeff"]

        self.clip_gradients_by_norm = config["clip_gradients_by_norm"]
        self.standardize_returns = config["standardize_return"]

        self.network = build_actor_network(
            obs_shape=self.env.observation_space.shape,
            n_outputs=self.num_outputs,
            num_dim=config["num_dim_actor"],
            act_f=config["act_f_actor"],
            output_act_f="linear",
        )

        self.num_episodes = config["num_episodes"]
        self.lr = config["lr"]
        self.actor_optimizer = optimizers.Adam(self.lr, epsilon=1e-5)

        self.logdir = config["logdir"]
        self.use_tensorboard = config["use_tensorboard"]

        # Save the configuration file to a folder
        os.makedirs(self.logdir, exist_ok=True)
        yaml.dump(config, open(os.path.join(self.logdir, "config.yaml"), "w"))

        # setup tensorboard writer
        if self.use_tensorboard:
            writer = tf.summary.create_file_writer(self.logdir)
            writer.set_as_default()

        self.total_episodes = 0

    def compute_action(self, obs):
        # get the network output
        action_dist_input = self.network(obs[None, ...])
        # create an action distribution class
        action_dist = self.action_dist_cls(action_dist_input)
        # Make a non deterministic sample or deterministic
        if self.explore:
            action = action_dist.sample()
        else:
            action = action_dist.deterministic_sample()

        # calculate log probability (not needed for REINFORCE)
        log_p = action_dist.log_p(action)
        # Return action, log_p and network output as numpy arrays
        return (
            action.numpy().squeeze(),
            log_p.numpy().squeeze(),
            action_dist_input.numpy().squeeze(),
        )

    def sample_trajectory(self):
        """
        Sample a single trajectory (episode) with a current policy
        :return: SampleBatch with a trajectory and score(total sum of rewards)
        """
        traj_dict = defaultdict(list)

        # reset the environment
        obs = self.env.reset()
        done = False
        score = 0
        while not done:
            action, log_p, action_dist_input = self.compute_action(obs)
            next_obs, reward, done, info = self.env.step(action)

            # Save experiences
            traj_dict[SampleBatch.OBS].append(obs)
            traj_dict[SampleBatch.ACTIONS].append(action)
            traj_dict[SampleBatch.DONES].append(done)
            traj_dict[SampleBatch.REWARDS].append(reward)

            # update score and observation
            score += reward
            obs = next_obs

        # Convert dictionary to SampleBatch
        sample_batch = SampleBatch(traj_dict)
        # Calculate returns
        sample_batch[SampleBatch.RETURNS] = discount_cumsum(
            sample_batch[SampleBatch.REWARDS], self.gamma
        )
        return sample_batch, score

    def train_op(self, obs, actions_old, returns):
        with tf.GradientTape() as tape:
            # -> forward pass
            action_dict_input = self.network(obs)
            action_dist = self.action_dist_cls(action_dict_input)
            log_p = action_dist.log_p(actions_old)

            # compute surrogate objective
            surrogate = log_p * returns
            # compute mean entropy
            mean_entropy = tf.reduce_mean(action_dist.entropy())
            # compute loss
            policy_loss = -tf.reduce_mean(surrogate)
            total_loss = policy_loss - self.entropy_coeff * mean_entropy

        # get gradients
        gradients = tape.gradient(total_loss, self.network.trainable_variables)
        # clip gradients by global norm (if neccessary)
        if self.clip_gradients_by_norm:
            gradients, global_norm = tf.clip_by_global_norm(
                gradients, self.clip_gradients_by_norm
            )
        # perform gradient descent
        self.actor_optimizer.apply_gradients(
            zip(gradients, self.network.trainable_variables)
        )
        return policy_loss, total_loss, mean_entropy

    def run(self, num_episodes=None, plot_stats=None, plot_period=1, history=None):
        # initialize plots
        if plot_stats:
            num_plots = len(plot_stats)
            fig, axs = plt.subplots(
                num_plots, 1, squeeze=False, figsize=(10, 5 * num_plots)
            )
            axs = axs.ravel()

        # initialize history dict
        history = history or {}
        history = defaultdict(list, history)

        num_episodes = num_episodes or self.num_episodes
        for i in range(num_episodes):
            # Keep track of current episode statistics
            stats = {}
            # sample a trajectory
            train_batch, score = self.sample_trajectory()

            # Get data from the training batch
            obs = train_batch[SampleBatch.OBS]
            actions_old = train_batch[SampleBatch.ACTIONS]
            returns = train_batch[SampleBatch.RETURNS].astype("float32")
            # normalize returns on the batch level
            returns = np_standardized(returns.squeeze())
            # perform gradient descent
            policy_loss, total_loss, entropy_bonus = self.train_op(
                obs, actions_old, returns
            )
            # record statistics
            stats["policy_loss"] = policy_loss.numpy().item()
            stats["entropy"] = entropy_bonus.numpy().item()
            stats["total_loss"] = total_loss.numpy().item()
            stats["score"] = score
            stats["steps_per_episode"] = len(train_batch)

            for k, v in stats.items():
                history[k].append(v)
                if self.use_tensorboard:
                    tf.summary.scalar(k, v, self.total_episodes)

            self.total_episodes += 1

            if plot_stats:
                if (i + 1) % plot_period == 0:
                    for ax, stat_name in zip(axs, plot_stats):
                        ax.clear()
                        # print(stat_name, len(history[stat_name]))

                        sns.lineplot(
                            x=np.arange(len(history[stat_name])),
                            y=history[stat_name],
                            ax=ax,
                        )

                        ax.set_title(stat_name)
                    display.display(fig)
                    display.clear_output(wait=True)

            else:
                print(
                    f"episode {i}/{self.num_episodes} | {stats}",
                )

        return history


def run_episode(env, agent):
    try:
        done = False
        score = 0

        obs = env.reset()
        while not done:
            action, __, __ = agent.compute_action(obs)
            obs, reward, done, info = env.step(action)
            score += reward
    except Exception as e:
        raise e
    finally:
        env.close()
    return score


if __name__ == "__main__":
    ROOT_DIR = os.path.join("Experiments", "pg_cartpole_continuous")

    agent = ReinforceAgent(
        config=dict(
            env_or_env_name=ContinuousCartPoleEnv(),
            # env_or_env_name='CartPole-v1',
            logdir=join("Experiments", "pg_cartpole_continuous"),
            use_tensorboard=False,
            explore=True,
            gamma=0.99,
            num_episodes=300,
            num_dim_actor=(32, 32),
            act_f_actor="tanh",
            entropy_coeff=1e-5,
            lr=0.0025,
            clip_gradients_by_norm=None,
            standardize_return=True,
        )
    )
    agent.run()
