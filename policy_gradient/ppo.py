import sys

import gym
import numpy as np
import tensorflow as tf
from memory.sample_batch import SampleBatch, discount_cumsum
from tensorboardX import SummaryWriter
from tensorflow.keras import layers, models, optimizers

logdir = 'test1'
writer = tf.summary.create_file_writer(logdir + "/metrics")
writer.set_as_default()

default_config = dict(
    explore=True,
    obs_shape=(8,),
    num_actions=2,
    continuous=False,
    clip_value=0.2,
    gamma=0.95,
    num_sgd_iter=20,
    num_epochs=20,
    sgd_minibatch_size=32,
    train_batch_size=128,
    num_dim_critic=(64, 64),
    act_f_critic="relu",
    num_dim_actor=(64, 64),
    act_f_actor="tanh",
    vf_share_layers=False,
    entropy_coeff=1e-5,
    lr_actor=5e-3,
    lr_critic=5e-4,
)


class PPOAgent:
    def __init__(self, config=None):
        config = config or {}
        self.config = config = {**default_config, **config}
        self.actor = build_actor_network(
            obs_shape=config["obs_shape"],
            num_actions=config["num_actions"],
            lr=config["lr_actor"],
            num_dim=config["num_dim_actor"],
            act_f=config["act_f_actor"],
        )

        self.critic = build_critic_network(
            obs_shape=config["obs_shape"],
            lr=config["lr_critic"],
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

    def _compute_action_discrete(self, obs):
        p = self.actor.predict(
            # [
            obs[None, :]
            # , np.zeros((1, 1)), np.zeros((1, self.num_actions))]
        )
        if self.explore:
            action = np.random.choice(self.num_actions, p=np.nan_to_num(p[0]))
        else:
            action = np.argmax(p[0])
        action_matrix = np.zeros(self.num_actions, dtype=np.float64)
        action_matrix[action] = 1
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
            if done:
                # episode_rewards[-1] = 0
                print(score)
                score = 0
                obs = env.reset()
                episode_id += 1
                rewards += discount_cumsum(
                    np.asarray(episode_rewards), self.gamma
                ).tolist()
                episode_rewards = []
        rewards += discount_cumsum(np.asarray(episode_rewards), self.gamma).tolist()
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
        optimizer = optimizers.Adam(self.lr_actor)

        class A:
            ...

        actor_history = A()
        actor_history.history = {"loss": []}

        while self.sgd_iters < self.num_sgd_iter:
            train_batch = self.get_batch()
            obs = train_batch[SampleBatch.OBS]
            action = train_batch[SampleBatch.ACTIONS]
            action_prob = train_batch[SampleBatch.ACTION_PROB]
            values = train_batch[SampleBatch.REWARDS]

            pred_values = self.critic.predict(obs)
            # pred_values = 0
            advantage = values - pred_values

            dataset = (
                tf.data.Dataset.from_tensor_slices(
                    (obs, advantage, action_prob, action)
                )
                .batch(self.sgd_minibatch_size)

            )
            for obs_batch, advantage_batch, action_prob_batch, action_batch in dataset:
                batch_loss = []
                with tf.GradientTape() as tape:
                    y_pred = self.actor(obs_batch)
                    y_true = action_batch

                    prob = tf.reduce_sum(y_true * y_pred, axis=-1)
                    old_prob = tf.reduce_sum(y_true * action_prob_batch, axis=-1)
                    prob_ratio = prob / (old_prob + 1e-10)
                    surrogate = prob_ratio * advantage_batch
                    surrogate_cliped = (
                        tf.clip_by_value(
                            prob_ratio, 1 - self.clip_value, 1 + self.clip_value
                        )
                        * advantage_batch
                    )
                    loss = -tf.reduce_mean(
                        tf.minimum(surrogate, surrogate_cliped)
                        + self.entropy_coeff * -(prob * tf.math.log(prob + 1e-10))
                    )
                    batch_loss.append(loss.numpy())


                actor_history.history["loss"].append(np.mean(batch_loss))
                gradient = tape.gradient(loss, self.actor.trainable_variables)
                for i in range(len(gradient)):
                    tf.summary.histogram(f'actor_gradient_{i}', gradient[i], step=self.total_epochs)
                for i in range(len(self.actor.trainable_variables)):
                    tf.summary.histogram(f'actor_weights_{i}', self.actor.trainable_variables[i], step=self.total_epochs)

                if True:
                    gradient, global_norm = tf.clip_by_global_norm(
                        gradient, 10.0
                    )
                optimizer.apply_gradients(zip(gradient, self.actor.trainable_variables))
                self.total_epochs += 1
            # self.writer.add_scalar(
            #     "action_0_prob", actor_history.history["loss"][-1], self.sgd_iters
            # )

            # actor_history = self.actor.fit(
            #     [obs, advantage, action_prob],
            #     [action],
            #     batch_size=self.sgd_minibatch_size,
            #     shuffle=True,
            #     epochs=self.num_epochs,
            #     verbose=0,
            # )
            critic_history = self.critic.fit(
                [obs],
                [values],
                batch_size=self.sgd_minibatch_size,
                shuffle=True,
                epochs=1,
                verbose=0,
            )
            tf.summary.scalar(
                "Actor loss", actor_history.history["loss"][-1], self.sgd_iters
            )
            tf.summary.scalar(
                "Critic loss", critic_history.history["loss"][-1], self.sgd_iters
            )
            print(
                self.sgd_iters,
                actor_history.history["loss"][-1],
                critic_history.history["loss"][-1],
            )
            self.sgd_iters += 1
            print(self.run_episode())

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
    obs_shape, loss="mse", lr=0.01, num_dim=(64, 64), act_f="tanh"
):
    state_input = layers.Input(shape=obs_shape)
    x = state_input
    for i, dim in enumerate(num_dim):
        x = layers.Dense(dim, activation=act_f, name=f"hidden_{i}")(x)

    out_value = layers.Dense(1, name="output")(x)

    model = models.Model(inputs=state_input, outputs=out_value, name="critic")
    model.compile(optimizer=optimizers.Adam(lr=lr, clipnorm=10.0, ), loss=loss)
    return model


def build_actor_network(
    obs_shape, num_actions, lr=0.01, num_dim=(64, 64), act_f="tanh"
):
    state_input = layers.Input(shape=obs_shape)
    # advantage = layers.Input(shape=(1,))
    # old_prediction = layers.Input(shape=(num_actions,))

    x = state_input
    for i, dim in enumerate(num_dim):
        x = layers.Dense(dim, activation=act_f, name=f"hidden_{i}")(x)

    out_actions = layers.Dense(num_actions, activation="softmax", name="output")(x)

    model = models.Model(inputs=state_input, outputs=out_actions, name="actor")
    # model.compile(optimizer=optimizers.Adam(lr=lr),
    #               loss=ppo_discrete_loss(
    #                   advantage=advantage,
    #                   old_prediction=old_prediction))
    model.summary()
    return model


from tensorflow.keras import backend as K


def ppo_discrete_loss(advantage, old_prediction, entropy_coeff=5e-3, clip_value=0.2):
    def loss(y_true, y_pred):
        prob = tf.reduce_sum(y_true * y_pred, axis=-1)
        old_prob = tf.reduce_sum(y_true * old_prediction, axis=-1)
        r = prob / (old_prob + 1e-10)
        loss = -tf.reduce_mean(
            tf.minimum(
                r * advantage,
                tf.clip_by_value(
                    r, clip_value_min=1 - clip_value, clip_value_max=1 + clip_value
                )
                * advantage,
            )
            + entropy_coeff * -(prob * tf.math.log(prob + 1e-10))
        )
        return loss

    return loss


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
    tf.keras.backend.set_floatx("float64")
    # env = gym.make("LunarLander-v2")
    env = gym.make("CartPole-v1")

    agent = PPOAgent(
        config=dict(
            obs_shape=env.observation_space.shape, num_actions=env.action_space.n
        )
    )
    agent.run()
