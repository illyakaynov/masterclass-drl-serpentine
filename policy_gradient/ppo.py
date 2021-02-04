import os
from collections import defaultdict

import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from gym.spaces import Box, Discrete
from memory.sample_batch import (
    SampleBatch,
    compute_advantages,
    np_standardized,
    tf_standardized,
)
from policy_gradient.cartpole_continuous import ClipActionsWrapper
from policy_gradient.networks import build_actor_network, build_critic_network
from policy_gradient.utils import (
    compute_entropy_discrete,
    compute_entropy_gaussian,
    compute_log_p_discrete,
    compute_log_p_gaussian,
    one_hot_encode,
)
from tensorflow.keras import optimizers

from policy_gradient.action_dist import CategoricalDistribution, GaussianDistribution

default_config = dict(
    logdir="default",
    explore=True,
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
    lr=0.01,
    vf_loss_coeff=1.0,
    vf_clip_param=10.0,
    clip_gradients_by_norm=None,
    use_critic=True,
    use_gae=True,
    standardize_advantages=True,
    gae_lambda=1.0,
    num_eval_episodes=1,
)

import yaml


class PPOAgent:
    def __init__(self, config=None):
        config = config or {}

        self.config = config = {**default_config, **config}

        self.logdir = config["logdir"]
        os.makedirs(self.logdir, exist_ok=True)
        yaml.dump(config, open(os.path.join(self.logdir, "config.yaml"), "w"))

        writer = tf.summary.create_file_writer(self.logdir)
        writer.set_as_default()

        env = config["env_or_env_name"]
        if isinstance(env, str):
            self.env = gym.make(env)

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
        self.num_sgd_iter = config["num_sgd_iter"]

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

        self.num_eval_episodes = config["num_eval_episodes"]

        self.actor = build_actor_network(
            obs_shape=self.env.observation_space.shape,
            n_outputs=self.num_outputs,
            num_dim=config["num_dim_actor"],
            act_f=config["act_f_actor"],
            output_act_f="linear" if self.continuous else "softmax",
        )

        self.critic = build_critic_network(
            obs_shape=self.env.observation_space.shape,
            num_dim=config["num_dim_critic"],
            act_f=config["act_f_critic"],
        )

        self.actor_optimizer = optimizers.Adam(self.lr, epsilon=1e-5)
        self.critic_optimizer = optimizers.Adam(self.lr, epsilon=1e-5)

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
        sample_batch = SampleBatch(traj_dict)

        if self.use_critic:
            sample_batch[SampleBatch.VF_PREDS] = agent.critic.predict(
                sample_batch[SampleBatch.OBS]
            )
        return sample_batch

    def sample_batch(self):
        samples = []
        num_samples = 0
        while num_samples < self.train_batch_size:
            trajectory = self.sample_trajectory()
            trajectory = compute_advantages(
                trajectory,
                last_r=0,
                gamma=self.gamma,
                lambda_=self.gae_lambda,
                use_critic=self.use_critic,
                use_gae=self.use_gae,
            )
            num_samples += trajectory.count
            samples.append(trajectory)
        return SampleBatch.concat_samples(samples)

    @tf.function
    def train_batch_critic(self, obs_batch, value_target_batch, old_value_pred_batch):
        with tf.GradientTape() as tape:
            value_fn_out = self.critic(obs_batch)
            vf_loss1 = tf.square(value_fn_out - value_target_batch)
            vf_clipped = old_value_pred_batch + tf.clip_by_value(
                value_fn_out - old_value_pred_batch,
                -self.vf_clip_param,
                self.vf_clip_param,
            )
            vf_loss2 = tf.square(vf_clipped - value_target_batch)
            vf_loss = tf.maximum(vf_loss1, vf_loss2)
            critic_loss = tf.reduce_mean(vf_loss) * self.vf_loss_coeff

        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        if self.clip_gradients_by_norm:
            critic_gradients, global_norm = tf.clip_by_global_norm(
                critic_gradients, self.clip_gradients_by_norm
            )
        self.critic_optimizer.apply_gradients(
            zip(critic_gradients, self.critic.trainable_variables)
        )

        # for i in range(len(self.actor.trainable_variables)):
        #     tf.summary.histogram(
        #         f"actor_weights_{i}",
        #         self.actor.trainable_variables[i],
        #         step=self.total_epochs,
        #     )

        return critic_loss

    @tf.function
    def train_actor_batch(
        self, obs_batch, action_old_batch, action_old_log_p_batch, advantage_batch
    ):
        with tf.GradientTape() as tape:
            action_dist_input = self.actor(obs_batch)
            action_dist = self.action_dist_cls(action_dist_input)
            log_p = action_dist.log_p(action_old_batch)

            prob_ratio = tf.exp(log_p - tf.squeeze(action_old_log_p_batch))
            advantage_batch = tf_standardized(tf.squeeze(advantage_batch))
            surrogate = prob_ratio * advantage_batch
            surrogate_cliped = (
                K.clip(prob_ratio, 1 - self.clip_value, 1 + self.clip_value)
                * advantage_batch
            )

            entropy_bonus = tf.reduce_mean(action_dist.entropy())
            actor_loss = -(
                tf.reduce_mean(tf.minimum(surrogate, surrogate_cliped))
                + self.entropy_coeff * entropy_bonus
            )

        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        if self.clip_gradients_by_norm:
            actor_gradients, global_norm = tf.clip_by_global_norm(
                actor_gradients, self.clip_gradients_by_norm
            )
        self.actor_optimizer.apply_gradients(
            zip(actor_gradients, self.actor.trainable_variables)
        )

        # for i in range(len(actor_gradients)):
        #     tf.summary.histogram(
        #         f"actor_gradient_{i}",
        #         actor_gradients[i],
        #         step=self.total_epochs,
        #     )

        return actor_loss, entropy_bonus

    def run(self):

        history = defaultdict(list)

        while self.sgd_iters < self.num_sgd_iter:
            epoch_actor_loss = []
            epoch_critic_loss = []
            epoch_entropy = []

            train_batch = self.sample_batch()

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

                critic_loss = self.train_batch_critic(
                    obs_batch, value_target_batch, old_value_pred_batch
                )

                actor_loss, entropy_bonus = self.train_actor_batch(
                    obs_batch, action_old_batch, action_old_log_p_batch, advantage_batch
                )

                epoch_critic_loss.append(critic_loss)
                epoch_actor_loss.append(actor_loss)
                epoch_entropy.append(entropy_bonus)

            history["actor_loss"].append(np.mean(epoch_actor_loss))
            history["critic_loss"].append(np.mean(epoch_critic_loss))
            history["entropy"].append(np.mean(epoch_entropy))

            tf.summary.scalar("Actor loss", history["actor_loss"][-1], self.sgd_iters)
            tf.summary.scalar("Critic loss", history["critic_loss"][-1], self.sgd_iters)
            tf.summary.scalar("Entropy", history["entropy"][-1], self.sgd_iters)

            print(
                self.sgd_iters,
                history["actor_loss"][-1],
                history["critic_loss"][-1],
            )
            self.sgd_iters += 1

            val_score = np.mean(
                [
                    run_episode(self.env, self, monitor=True, logdir=self.logdir)
                    for i in range(self.num_eval_episodes)
                ]
            )
            tf.summary.scalar("Validation Score", val_score, self.sgd_iters)
            history["score"].append(val_score)
            print(val_score)

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
        action, __, __ = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        score += reward
    env.close()
    return score


if __name__ == "__main__":
    agent = PPOAgent(
        config=dict(
            env_or_env_name="LunarLanderContinuous-v2",
            logdir=os.path.join("lunarlander_continuous", "action_dist"),
            explore=True,
            continuous=False,
            clip_value=0.2,
            gamma=0.99,
            num_sgd_iter=100,
            num_epochs=30,
            sgd_minibatch_size=128,
            train_batch_size=4000,
            num_dim_critic=[64, 64, 128],
            act_f_critic="relu",
            num_dim_actor=[64, 64, 128],
            act_f_actor="tanh",
            vf_share_layers=False,
            entropy_coeff=1e-3,
            lr=0.00025,
            vf_loss_coeff=1.0,
            clip_gradients_by_norm=None,
        )
    )
    from gym.wrappers import Monitor

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
