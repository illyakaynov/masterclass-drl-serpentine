import os
from collections import defaultdict
from os.path import join

import gym
import numpy as np
import seaborn as sns
import tensorflow as tf
import yaml
from gym.spaces import Box
from IPython.core import display
from matplotlib import pyplot as plt
from policy_gradient.action_dist import CategoricalDistribution, GaussianDistribution
from policy_gradient.cartpole_continuous import (
    ClipActionsWrapper,
    ContinuousCartPoleEnv,
)
from policy_gradient.memory.sample_batch import (
    SampleBatch,
    compute_advantages,
    tf_standardized,
)
from policy_gradient.networks import build_actor_network, build_critic_network
from tensorflow.keras import optimizers

default_config = dict(
    # Folder where to save files related to the run
    logdir=join("Experiments", "ppo_default"),
    # True to use tensorboard for logging
    use_tensorboard=False,
    # True to perform non-deterministic during an episode
    explore=True,
    # PPO clip parameter
    clip_value=0.2,
    # Discount for rewards
    gamma=0.99,
    # Number of iterations
    num_iter=6,
    # Number of epoch per iteration
    num_epochs=20,
    # Size of the training batch
    train_batch_size=4000,
    # Size of the mini-batch
    sgd_minibatch_size=128,
    # Dimensions of the dense layers of the actor network
    num_dim_actor=(64, 64),
    # Activation function of the dense layers of the actor network
    act_f_actor="tanh",
    # Dimensions of the dense layers of the critic network
    num_dim_critic=(64, 64),
    # Activation function of the dense layers of the critic network
    act_f_critic="tanh",
    # Entropy coefficient, used to control exploration
    entropy_coeff=1e-5,
    # Learning rate
    lr=0.01,
    # Coefficient of the value function loss.
    vf_loss_coeff=1.0,
    # Clip param for the value function. Note that this is sensitive to the
    # scale of the rewards. If your expected V is large, increase this.
    vf_clip_param=10.0,
    # If specified, clip the global norm of gradients by this amount.
    clip_gradients_by_norm=None,
    # Should use a critic as a baseline (otherwise don't use value baseline;
    # required for using GAE).
    use_critic=True,
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    use_gae=True,
    # The GAE(lambda) parameter.
    gae_lambda=1.0,
    # True to normalize the returns (0 mean, 1 variance) on the mini-batch level
    standardize_advantages=True,
)


class PPOAgent:
    def __init__(self, config=None):
        config = config or {}

        self.config = config = {**default_config, **config}

        self.env = config["env_or_env_name"]
        if isinstance(self.env, str):
            self.env = gym.make(self.env)

        self.continuous = True if isinstance(self.env.action_space, Box) else False
        if self.continuous:
            self.num_outputs = self.env.action_space.shape[0] * 2
            self.action_dist_cls = GaussianDistribution
            self.env = ClipActionsWrapper(self.env)
        else:
            self.num_outputs = self.env.action_space.n
            self.action_dist_cls = CategoricalDistribution

        self.explore = config["explore"]
        self.train_batch_size = config["train_batch_size"]
        self.sgd_minibatch_size = config["sgd_minibatch_size"]
        self.num_epochs = config["num_epochs"]
        self.num_iter = config["num_iter"]

        self.gamma = config["gamma"]
        self.lr = config["lr"]
        self.vf_loss_coeff = config["vf_loss_coeff"]
        self.vf_clip_param = config["vf_clip_param"]
        self.clip_value = config["clip_value"]
        self.entropy_coeff = config["entropy_coeff"]

        self.sgd_iters = 0
        self.total_epochs = 0

        self.clip_gradients_by_norm = config["clip_gradients_by_norm"]
        self.use_critic = config["use_critic"]
        self.use_gae = config["use_gae"]
        self.gae_lambda = config["gae_lambda"]
        self.standardize_advantages = config["standardize_advantages"]

        self.actor = build_actor_network(
            obs_shape=self.env.observation_space.shape,
            n_outputs=self.num_outputs,
            num_dim=config["num_dim_actor"],
            act_f=config["act_f_actor"],
            output_act_f="linear",
        )

        self.critic = build_critic_network(
            obs_shape=self.env.observation_space.shape,
            num_dim=config["num_dim_critic"],
            act_f=config["act_f_critic"],
        )

        self.actor_optimizer = optimizers.Adam(self.lr, epsilon=1e-5)
        self.critic_optimizer = optimizers.Adam(self.lr, epsilon=1e-5)

        self.logdir = config["logdir"]
        self.use_tensorboard = config["use_tensorboard"]

        os.makedirs(self.logdir, exist_ok=True)
        yaml.dump(config, open(os.path.join(self.logdir, "config.yaml"), "w"))

        if self.use_tensorboard:
            writer = tf.summary.create_file_writer(self.logdir)
            writer.set_as_default()

        self.total_iters = 0

    def compute_action(self, obs):
        action_dist_input = self.actor(obs[None, ...])
        action_dist = self.action_dist_cls(action_dist_input)
        if self.explore:
            action = action_dist.sample()
        else:
            action = action_dist.deterministic_sample()

        log_p = action_dist.log_p(action)
        return (
            action.numpy().squeeze(),
            log_p.numpy().squeeze(),
            action_dist_input.numpy().squeeze(),
        )

    def sample_trajectory(self):
        traj_dict = defaultdict(list)

        obs = self.env.reset()
        done = False
        score = 0
        # Run one episode
        while not done:
            action, log_p, action_dist_input = self.compute_action(obs)
            next_obs, reward, done, info = self.env.step(action)

            traj_dict[SampleBatch.OBS].append(obs)
            traj_dict[SampleBatch.ACTIONS].append(action)
            traj_dict[SampleBatch.DONES].append(done)
            traj_dict[SampleBatch.REWARDS].append(reward)
            traj_dict[SampleBatch.ACTION_DIST_INPUTS].append(action_dist_input)
            traj_dict[SampleBatch.ACTION_LOGP].append(log_p)

            obs = next_obs
            score += reward
        # Convert simple dictionary to SampleBatch
        sample_batch = SampleBatch(traj_dict)
        # Compute Value Estimates
        sample_batch[SampleBatch.VF_PREDS] = self.critic.predict(
            sample_batch[SampleBatch.OBS]
        )
        # Compute Advantages
        sample_batch = compute_advantages(
            sample_batch,
            last_r=0,
            gamma=self.gamma,
            lambda_=self.gae_lambda,
            use_critic=self.use_critic,
            use_gae=self.use_gae,
        )
        return sample_batch, score

    def sample_batch(self):
        samples = []
        scores = []
        num_episodes = 0
        num_steps = 0
        while num_steps < self.train_batch_size:
            trajectory, score = self.sample_trajectory()
            num_steps += trajectory.count
            num_episodes += 1
            scores.append(score)
            samples.append(trajectory)

        batch_stats = dict(
            num_episodes=num_episodes,
            num_steps=num_steps,
            mean_steps_per_episode=num_steps / num_episodes,
            mean_score=np.mean(scores),
            min_score=np.min(scores),
            max_score=np.max(scores),
        )

        return SampleBatch.concat_samples(samples), batch_stats

    @tf.function
    def train_op_critic(self, obs_batch, value_target_batch, old_value_pred_batch):
        with tf.GradientTape() as tape:
            # Get current value estimates
            value_fn_out = self.critic(obs_batch)
            # Compute the squared difference
            vf_loss1 = tf.square(value_fn_out - value_target_batch)
            # Compute clipped vf
            vf_clipped = old_value_pred_batch + tf.clip_by_value(
                value_fn_out - old_value_pred_batch,
                -self.vf_clip_param,
                self.vf_clip_param,
            )
            vf_loss2 = tf.square(vf_clipped - value_target_batch)
            # Since this value already has appropriate sign take the maximum
            vf_loss = tf.maximum(vf_loss1, vf_loss2)
            # Calculate critic loss
            critic_loss = tf.reduce_mean(vf_loss) * self.vf_loss_coeff
        # Get the gradients
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        # Clip gradients by the global l2 norm
        if self.clip_gradients_by_norm:
            critic_gradients, global_norm = tf.clip_by_global_norm(
                critic_gradients, self.clip_gradients_by_norm
            )
        # Perform Gradient Descent
        self.critic_optimizer.apply_gradients(
            zip(critic_gradients, self.critic.trainable_variables)
        )
        return critic_loss

    @tf.function
    def train_op_actor(
        self, obs_batch, action_old_batch, action_old_log_p_batch, advantage_batch
    ):
        with tf.GradientTape() as tape:
            # Inference the actor network
            action_dist_input = self.actor(obs_batch)
            # Create action distribution
            action_dist = self.action_dist_cls(action_dist_input)
            # Calculate log probability of the old actions under current policy
            log_p = action_dist.log_p(action_old_batch)
            # calculate the importance sampling probability ratio r(\theta)
            prob_ratio = tf.exp(log_p - tf.squeeze(action_old_log_p_batch))
            # Normalize the advantages (zero mean, unit variance)
            advantage_batch = tf.squeeze(advantage_batch)
            if self.standardize_advantages:
                advantage_batch = tf_standardized(advantage_batch)
            # Compute surrogate objective
            surrogate = prob_ratio * advantage_batch
            # Compute clipped surrogate objective
            surrogate_cliped = (
                tf.clip_by_value(prob_ratio, 1 - self.clip_value, 1 + self.clip_value)
                * advantage_batch
            )
            # Compute entropy of action distribution of the current policy
            mean_entropy = tf.reduce_mean(action_dist.entropy())
            # take a minimum between clipped and un-clipped surrogate objective
            # Take a negative since we performing Gradient Descent
            policy_loss = -tf.reduce_mean(tf.minimum(surrogate, surrogate_cliped))
            # Adjust mean entropy with a coefficient and subtract from the policy loss
            actor_loss = policy_loss - self.entropy_coeff * mean_entropy
        # take the gradients
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        # clip gradients by norm
        if self.clip_gradients_by_norm:
            actor_gradients, global_norm = tf.clip_by_global_norm(
                actor_gradients, self.clip_gradients_by_norm
            )
        # perform Gradient Descent
        self.actor_optimizer.apply_gradients(
            zip(actor_gradients, self.actor.trainable_variables)
        )

        return policy_loss, mean_entropy, actor_loss

    def run(self, num_iter=None, plot_stats=None, plot_period=1, history=None):
        # initialize plots
        plot_stats = plot_stats or []
        if plot_stats:
            num_plots = len(plot_stats)
            fig, axs = plt.subplots(
                num_plots, 1, squeeze=False, figsize=(10, 5 * num_plots)
            )
            axs = axs.ravel()

        # initialize history dict
        history = history or {}
        history = defaultdict(list, history)

        total_steps = history.get("total_steps", [0])[-1]
        total_episodes = history.get("total_episodes", [0])[-1]

        num_iter = num_iter or self.num_iter
        for i in range(num_iter):
            # Store statistics of the current update step
            stats = defaultdict(list)
            # sample a training batch
            train_batch, train_batch_stats = self.sample_batch()
            # create a datatset
            obs = train_batch[SampleBatch.OBS]
            actions_old = train_batch[SampleBatch.ACTIONS]
            action_old_log_p = train_batch[SampleBatch.ACTION_LOGP]
            advantages = train_batch[SampleBatch.ADVANTAGES].astype("float32")
            value_targets = train_batch[SampleBatch.VALUE_TARGETS].astype("float32")
            old_value_pred = train_batch[SampleBatch.VF_PREDS].astype("float32")
            dataset = (
                tf.data.Dataset.from_tensor_slices(
                    (
                        obs,
                        advantages,
                        action_old_log_p,
                        actions_old,
                        value_targets,
                        old_value_pred,
                    )
                )
                .batch(self.sgd_minibatch_size, drop_remainder=True)
                .shuffle(1)
            )
            for (
                obs_batch,
                advantage_batch,
                action_old_log_p_batch,
                action_old_batch,
                value_target_batch,
                old_value_pred_batch,
            ) in dataset:
                # if critic is not used critic loss is zero
                critic_loss = tf.constant([0])
                # update critic
                if self.use_critic:
                    critic_loss = self.train_op_critic(
                        obs_batch, value_target_batch, old_value_pred_batch
                    )
                # update actor
                policy_loss, mean_entropy, actor_loss = self.train_op_actor(
                    obs_batch, action_old_batch, action_old_log_p_batch, advantage_batch
                )
                stats["policy_loss"].append(policy_loss.numpy().item())
                stats["mean_entropy"].append(mean_entropy.numpy().item())
                stats["actor_loss"].append(actor_loss.numpy().item())
                stats["critic_loss"].append(critic_loss.numpy().item())

            mean_stats = {k: np.mean(v) for k, v in stats.items()}
            mean_stats = dict(**mean_stats, **train_batch_stats)
            # record total steps per game and episodes per iteration
            total_steps += mean_stats["num_steps"]
            total_episodes += mean_stats["num_episodes"]
            mean_stats["total_steps"] = total_steps
            mean_stats["total_episodes"] = total_episodes

            for k, v in mean_stats.items():
                history[k].append(v)
                if self.use_tensorboard:
                    tf.summary.scalar(k, v, self.total_iters)

            if plot_stats:
                if (i + 1) % plot_period == 0:
                    for ax, stat_name in zip(axs, plot_stats):
                        ax.clear()
                        if isinstance(stat_name, str):
                            stat_name = [stat_name]
                        for s in stat_name:
                            sns.lineplot(
                                x=np.arange(len(history[s])),
                                y=history[s],
                                ax=ax,
                            )
                        ax.set_title(stat_name)
                    display.display(fig)
                    display.clear_output(wait=True)
            else:
                print(
                    f"Iteration: {i+1}/{self.num_iter} | {mean_stats}",
                )

            self.total_iters += 1

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
    agent = PPOAgent(
        config=dict(
            # env_or_env_name="LunarLanderContinuous-v2",
            # env_or_env_name="LunarLander-v2",
            env_or_env_name=ContinuousCartPoleEnv(),
            logdir=join("Experiments", "ppo_default"),
            use_tensorboard=False,
            explore=True,
            clip_value=0.2,
            gamma=0.99,
            num_iter=20,
            num_epochs=20,
            train_batch_size=1024,
            sgd_minibatch_size=32,
            num_dim_actor=(
                32,
                32,
            ),
            act_f_actor="tanh",
            num_dim_critic=(
                32,
                32,
            ),
            act_f_critic="tanh",
            entropy_coeff=1e-3,
            lr=0.00025,
            vf_loss_coeff=1.0,
            vf_clip_param=10.0,
            clip_gradients_by_norm=0.5,
            use_critic=True,
            use_gae=True,
            gae_lambda=1.0,
            standardize_advantages=True,
        )
    )
    history = agent.run()
