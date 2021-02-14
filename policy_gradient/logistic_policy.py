from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.core import display


class LogisticAgent:
    def __init__(self, params, lr, gamma):
        # Initialize paramters, learning rate and discount factor

        self.params = params
        self.lr = lr
        self.gamma = gamma

    def logistic(self, y):
        # definition of logistic function

        return 1 / (1 + np.exp(-y))

    def action_probs(self, x):
        # returns probabilities of two actions

        y = np.dot(x, self.params)
        prob0 = self.logistic(y)

        return np.array([prob0, 1 - prob0])

    def compute_action(self, x):
        # sample an action in proportion to probabilities

        probs = self.action_probs(x)
        action = np.random.choice([0, 1], p=probs)

        return action, probs[action]

    def grad_log_p(self, x):
        # calculate grad-log-probs

        y = np.dot(x, self.params)
        grad_log_p0 = x - x * self.logistic(y)
        grad_log_p1 = -x * self.logistic(y)

        return grad_log_p0, grad_log_p1

    def grad_log_p_dot_rewards(self, grad_log_p, discounted_rewards):
        # dot grads with future rewards for each action in episode
        return np.dot(grad_log_p.T, discounted_rewards)

    def discount_rewards(self, rewards):
        # calculate temporally adjusted, discounted rewards

        discounted_rewards = np.zeros(len(rewards))
        cumulative_rewards = 0
        for i in reversed(range(0, len(rewards))):
            cumulative_rewards = cumulative_rewards * self.gamma + rewards[i]
            discounted_rewards[i] = cumulative_rewards

        return discounted_rewards

    def update(self, rewards, obs, actions):
        # calculate gradients for each action over all observations
        grad_log_p = np.array(
            [self.grad_log_p(ob)[action] for ob, action in zip(obs, actions)]
        )

        assert grad_log_p.shape == (len(obs), 4)

        # calculate temporaly adjusted, discounted rewards
        discounted_rewards = self.discount_rewards(rewards)

        # gradients times rewards
        grad_surrogate_objective = self.grad_log_p_dot_rewards(grad_log_p, discounted_rewards)

        # gradient ascent on parameters
        self.params += self.lr * grad_surrogate_objective


def run_episode(env, agent, render=False):

    observation = env.reset()
    totalreward = 0

    observations = []
    actions = []
    rewards = []
    probs = []

    done = False

    while not done:
        if render:
            env.render()

        observations.append(observation)

        action, prob = agent.compute_action(observation)
        observation, reward, done, info = env.step(action)

        totalreward += reward
        rewards.append(reward)
        actions.append(action)
        probs.append(prob)

    return (
        totalreward,
        np.array(rewards),
        np.array(observations),
        np.array(actions),
        np.array(probs),
    )


def train(
    env,
    params,
    lr,
    gamma,
    agent_cls,
    MAX_EPISODES=1000,
    plot_stats=None,
    plot_period=1,
    seed=None,
    evaluate=False,
    video_folder="",
):
    plot_stats = plot_stats or []
    if plot_stats:
        num_plots = len(plot_stats)
        fig, axs = plt.subplots(
            num_plots, 1, squeeze=False, figsize=(10, 5 * num_plots)
        )
        axs = axs.ravel()

    history = defaultdict(list)

    # initialize environment and the agent
    if seed is not None:
        env.seed(seed)
    episode_rewards = []
    agent = agent_cls(params, lr, gamma)

    # train until MAX_EPISODES
    for i in range(MAX_EPISODES):
        stats = {}
        # run a single episode
        total_reward, rewards, observations, actions, probs = run_episode(env, agent)
        # keep track of episode rewards
        episode_rewards.append(total_reward)
        # update agent
        agent.update(rewards, observations, actions)
        stats["score"] = total_reward

        for k, v in stats.items():
            history[k].append(v)

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
                f"episode {i}/{MAX_EPISODES} | {stats}",
            )

    return episode_rewards, agent


if __name__ == "__main__":
    import gym
    import gym.wrappers

    GLOBAL_SEED = 0
    np.random.seed(GLOBAL_SEED)
    env = gym.make("CartPole-v0")

    episode_rewards, policy = train(
        env,
        params=np.random.rand(4),
        lr=0.002,
        gamma=0.99,
        agent_cls=LogisticAgent,
        MAX_EPISODES=2000,
        seed=GLOBAL_SEED,
        plot_stats=['score'],
        plot_period=100,
    )
