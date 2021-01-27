import sys

import gym
import numpy as np
import tensorflow as tf
from memory.sample_batch import SampleBatch, compute_advantages, standardized

# from tensorboardX import SummaryWriter
from tensorflow.keras import layers, models, optimizers

import tensorflow.keras.backend as K

from collections import defaultdict


logdir = "test1"
writer = tf.summary.create_file_writer(logdir + "/metrics")
writer.set_as_default()


def normalize(x):
    x = x.flatten()
    return ((x - x.mean()) / (x.std() + 1e-10)).flatten()


def one_hot_encode(x, max_x):
    x = x.flatten()
    x_one_hot = np.zeros((x.size, max_x))
    x_one_hot[np.arange(x.size), x] = 1
    return x_one_hot


def calculate_log_p_continuous(x, mean, log_std):
    std = np.exp(log_std)
    return (
        -0.5
        * tf.reduce_sum(tf.math.square((tf.cast(x, tf.float32) - mean) / std), axis=1)
        - 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[1], tf.float32)
        - tf.reduce_sum(log_std, axis=1)
    )


def calculate_log_p_discrete(action_one_hot, action_prob):
    return -tf.keras.losses.categorical_crossentropy(
        action_one_hot, action_prob, from_logits=False
    )


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
    lr=0.01,
    vf_loss_coeff=1.0,
    vf_clip_param=10.0,

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
        self.lr = config["lr"]
        self.vf_loss_coeff = config["vf_loss_coeff"]
        self.vf_clip_param = config["vf_clip_param"]
        self.clip_value = config["clip_value"]
        self.entropy_coeff = config["entropy_coeff"]

        self.sgd_iters = 0
        self.total_epochs = 0

        self.clip_gradients_by_norm = config["clip_gradients_by_norm"]

        self.actor_optimizer = optimizers.Adam(self.lr)
        self.critic_optimizer = optimizers.Adam(self.lr)

    def _compute_action_discrete(self, obs):
        action_probs = self.actor.predict(obs[None, :])
        if self.explore:
            action = np.random.choice(
                self.num_actions, p=np.nan_to_num(action_probs[0])
            )
        else:
            action = np.argmax(action_probs[0])
        return action, action_probs

    def get_action_continuous(self, obs):
        p = self.actor.predict(obs[None, ...])
        action = action_probs = p[0] + np.random.normal(
            loc=0, scale=1.0, size=p[0].shape
        )

        return action, action_probs

    def compute_action(self, obs):
        if self.continuous:
            ...
        else:
            action, action_probs = self._compute_action_discrete(obs)
        # print(p)
        return action, action_probs

    def run(self):

        history = defaultdict(list)

        while self.sgd_iters < self.num_sgd_iter:
            epoch_actor_loss = []
            epoch_critic_loss = []

            train_batch = sample_batch(env, self, self.train_batch_size)
            obs = train_batch[SampleBatch.OBS]
            actions_old = train_batch[SampleBatch.ACTIONS]
            actions_old = one_hot_encode(actions_old, self.num_actions)

            action_prob = train_batch[SampleBatch.ACTION_PROB]
            action_old_log_p = calculate_log_p_discrete(actions_old, action_prob)
            advantages = train_batch[SampleBatch.ADVANTAGES].astype("float32")
            value_targets = train_batch[SampleBatch.VALUE_TARGETS].astype("float32")
            old_value_pred = train_batch[SampleBatch.VF_PREDS].astype("float32")
            # pred_values = 0

            dataset = (
                tf.data.Dataset.from_tensor_slices(
                    (obs, advantages, action_old_log_p, actions_old, value_targets, old_value_pred)
                )
                .batch(self.sgd_minibatch_size)
                .shuffle(1)
            )
            for (
                obs_batch,
                advantage_batch,
                action_old_log_p_batch,
                action_old_batch,
                value_target_batch,
                old_value_pred_batch
            ) in dataset:

                with tf.GradientTape() as tape:
                    value_fn_out = self.critic(obs_batch)
                    vf_loss1 = tf.square(value_fn_out - value_target_batch)
                    vf_clipped = old_value_pred_batch + tf.clip_by_value(
                        value_fn_out - old_value_pred_batch, -self.vf_clip_param,
                        self.vf_clip_param)
                    vf_loss2 = tf.square(vf_clipped - value_target_batch)
                    vf_loss = tf.maximum(vf_loss1, vf_loss2)
                    critic_loss = tf.reduce_mean(vf_loss) * self.vf_loss_coeff

                critic_gradients = tape.gradient(
                    critic_loss, self.critic.trainable_variables
                )
                if self.clip_gradients_by_norm:
                    critic_gradients, global_norm = tf.clip_by_global_norm(
                        critic_gradients, self.clip_gradients_by_norm
                    )
                self.critic_optimizer.apply_gradients(
                    zip(critic_gradients, self.critic.trainable_variables)
                )
                epoch_critic_loss.append(critic_loss)

                with tf.GradientTape() as tape:
                    action_prob_pred = self.actor(obs_batch)
                    log_p = calculate_log_p_discrete(action_old_batch, action_prob_pred)
                    prob_ratio = tf.exp(log_p - action_old_log_p_batch)
                    advantage_batch = tf.squeeze(advantage_batch)
                    surrogate = prob_ratio * advantage_batch
                    surrogate_cliped = (
                        K.clip(prob_ratio, 1 - self.clip_value, 1 + self.clip_value)
                        * advantage_batch
                    )
                    # entropy = - sum_x x * log(x)
                    entropy_bonus = tf.reduce_mean(
                        -tf.reduce_sum(
                            (action_prob_pred * tf.math.log(action_prob_pred + 1e-10)),
                            axis=1,
                        )
                    )
                    actor_loss = -(
                        tf.reduce_mean(tf.minimum(surrogate, surrogate_cliped))
                        + self.entropy_coeff * entropy_bonus
                    )

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

                # for i in range(len(actor_gradients)):
                #     tf.summary.histogram(
                #         f"actor_gradient_{i}",
                #         actor_gradients[i],
                #         step=self.total_epochs,
                #     )
                # for i in range(len(self.actor.trainable_variables)):
                #     tf.summary.histogram(
                #         f"actor_weights_{i}",
                #         self.actor.trainable_variables[i],
                #         step=self.total_epochs,
                #     )
                tf.summary.scalar(
                    "entropy", tf.reduce_mean(entropy_bonus), step=self.total_epochs
                )
                # for i in range(len(prob)):
                #     tf.summary.scalar(f'action_prob_{i}', tf.reduce_mean(prob[i]), step=self.total_epochs)

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
            tf.summary.scalar("Validation Reward", val_score, self.sgd_iters)
            history["score"].append(val_score)
            print(val_score)

        return history

    def run_episode(self):
        done = False
        score = 0
        env = Monitor(gym.make("LunarLander-v2"), logdir, video_callable=lambda x: True, force=True)

        obs = env.reset()
        while not done:
            action, __ = self.compute_action(obs)
            obs, reward, done, info = env.step(action)
            score += reward
        env.close()
        return score


def build_critic_network(obs_shape, num_dim=(64, 64), act_f="tanh"):
    state_input = layers.Input(shape=obs_shape, dtype=tf.float32)
    x = state_input
    for i, dim in enumerate(num_dim):
        x = layers.Dense(dim, activation=act_f, name=f"hidden_{i}")(x)

    out_value = layers.Dense(1, name="output")(x)

    model = models.Model(inputs=state_input, outputs=out_value, name="critic")
    model.summary()
    return model


def build_actor_network(
    obs_shape, num_actions, num_dim=(64, 64), act_f="tanh", output_act_f="softmax"
):
    state_input = layers.Input(shape=obs_shape, dtype=tf.float32)

    x = state_input
    for i, dim in enumerate(num_dim):
        x = layers.Dense(dim, activation=act_f, name=f"hidden_{i}", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))(x)

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


def sample_trajectory(env, agent, fetch_values=True):
    traj_dict = defaultdict(list)

    sum_reward = 0
    obs = env.reset()
    done = False
    num_steps = 0
    while not done:
        action, action_prob = agent.compute_action(obs)
        next_obs, reward, done, info = env.step(action)

        traj_dict[SampleBatch.OBS].append(obs)
        traj_dict[SampleBatch.ACTIONS].append(action)
        traj_dict[SampleBatch.DONES].append(done)
        traj_dict[SampleBatch.ACTION_PROB].append(action_prob)
        traj_dict[SampleBatch.REWARDS].append(reward)

        sum_reward += reward
        obs = next_obs
        num_steps += 1
    sample_batch = SampleBatch(traj_dict)

    if fetch_values:
        sample_batch[SampleBatch.VF_PREDS] = agent.critic.predict(
            sample_batch[SampleBatch.OBS]
        )
    return sample_batch


def sample_batch(env, agent, training_batch_size):
    samples = []
    num_samples = 0
    while num_samples < training_batch_size:
        trajectory = sample_trajectory(env, agent)
        trajectory = compute_advantages(
            trajectory,
            last_r=0,
            gamma=0.99,
            lambda_=1.,
            use_critic=True,
            use_gae=True,
            standardize_advantages=True,
        )
        num_samples += trajectory.count
        samples.append(trajectory)
    return SampleBatch.concat_samples(samples)


if __name__ == "__main__":

    from gym.wrappers import TransformObservation

    # tf.keras.backend.set_floatx("float64")
    env = gym.make("LunarLander-v2")
    # env = TransformObservation(
    #     gym.make("CartPole-v1")
        # gym.make("LunarLander-v2")
        # ,
        # f=lambda x: x.astype(np.float32),
    # )

    agent = PPOAgent(
        config=dict(
            obs_shape=env.observation_space.shape,
            num_actions=env.action_space.n,
            explore=True,
            continuous=False,
            clip_value=0.2,
            gamma=0.99,
            num_sgd_iter=100,
            num_epochs=30,
            sgd_minibatch_size=128,
            train_batch_size=1024,
            num_dim_critic=(512, 256, 64),
            act_f_critic="relu",
            num_dim_actor=(512, 256, 64),
            act_f_actor="relu",
            vf_share_layers=False,
            entropy_coeff=1e-3,
            lr=0.00025,
            vf_loss_coeff=1.,
            clip_gradients_by_norm=None,
        )
    )
    from gym.wrappers import Monitor
    logdir = "moonlande_2"

    writer = tf.summary.create_file_writer(logdir + "/batch_size_128_lambda_1_epochs_30")
    writer.set_as_default()
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
