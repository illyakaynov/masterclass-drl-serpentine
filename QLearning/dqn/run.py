from IPython.core import display

import time
from os.path import join

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# Define loop for one episode
def run_episode(env, agent, train=False, max_steps=100_000, render=False, sleep=0.1):
    """
    run one episode and report statistics
    :param env: (gym.Environment)
    :param agent: (object) an object that implements compute_action(), update() and finalize_episode() methods
    :param train: (bool) True - train the network, False - run only for inference
    :param max_steps: (int) max steps per episode
    :param render: (bool) whether to show the agent playing
    :param sleep: (float) delay between consequent frames, use only when render=True
    :return: (dict) dictionary containing statistics about the run
    """
    is_terminal = False
    score = 0
    steps = 0
    start_timer = time.time()

    current_state = env.reset()

    if render:
        img = plt.imshow(env.render(mode='rgb_array'))  # only call this once

    while (not is_terminal) and (steps < max_steps):
        # get action from the agent

        action = agent.compute_action(current_state)

        # advance one step in the environment
        next_state, reward, is_terminal, info = env.step(action)

        # update score
        score += reward

        # Clip the agent's reward - improves stability
        reward = np.clip(reward, -1, 1)

        if train:
            # save the transition
            agent.update(current_state, action, reward, next_state, is_terminal)

        # update current state
        current_state = next_state

        if render:
            # render the frame and pause
            img.set_data(env.render(mode='rgb_array'))  # just update the data
            display.display(plt.gcf())
            display.clear_output(wait=True)
            time.sleep(sleep)

        steps += 1
        # print(steps)

    # Call to clear internal agent statistics (reward, return, etc.)
    stats = agent.finalize_episode()

    total_time = time.time() - start_timer
    return {'score': score,
            'steps_per_game': steps,
            'framerate': steps / (total_time + 1e-6),
            **stats
            }


# Define loop for multiple episodes
def run_experiment(env,
                   agent,
                   runs=100,
                   plot_stats=['score', 'steps_per_game', 'framerate', 'loss', 'epsilon'],
                   history={}):
    """

    :param env: (gym.Environment)
    :param agent: (object) an object that implements compute_action() and update() methods
    :param runs: (int) number of episodes
    :param plot_stats: (array_like) a list of elements to plot from a history
    :param history:  (dict) history object to be able to call this function
     multiple times without loosing the data about the training process
    :return: (dict) history
    """
    num_plots = len(plot_stats)
    fig, axs = plt.subplots(num_plots, 1, squeeze=False, figsize=(10, 5 * num_plots))
    axs = axs.ravel()

    if history is None:
        history = {}

    for i in range(runs):
        stats = run_episode(env, agent, train=True, render=False)

        #       Update history object
        for k, v in stats.items():
            if k not in history.keys():
                history[k] = []
            history[k].append(v)

        for ax, stat_name in zip(axs, plot_stats):
            ax.clear()
            sns.lineplot(x=np.arange(len(history[stat_name])), y=history[stat_name], ax=ax)
            ax.set_title(stat_name)
        display.display(fig)
        display.clear_output(wait=True)

    #         print(f"episode {i}/{runs} | {stats}")
    return history


def plot_history(histories,
                 plot_stats=['score', 'steps_per_game', 'framerate', 'loss', 'epsilon'],
                 names=None):
    num_plots = len(plot_stats)

    fig, axs = plt.subplots(num_plots, 1, squeeze=False, figsize=(10, 5 * num_plots))
    axs = axs.ravel()

    if isinstance(histories, dict):
        histories = [histories]

    if names is None:
        names = ['' for _ in range(len(histories))]

    for i, history in enumerate(histories):
        for ax, stat_name in zip(axs, plot_stats):
            # ax.clear()
            sns.lineplot(x=np.arange(len(history[stat_name])),
                         y=history[stat_name], ax=ax, label=names[i], alpha=0.7)
            ax.set_title(stat_name)

def load_history(path):
    import json
    with open(path, 'r') as f:
        history = json.loads(json.load(f).replace("'", '"'))
    return history


def save_history(history, path):
    import json
    json.dump(str(history), open(path), 'w')