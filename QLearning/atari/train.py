from dqn.agent import QAgent
import numpy as np
import gym
from IPython import display
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline
import time


# Define loop for one episode
def run_episode(env, agent, train=False, max_steps=100_000, render=False, sleep=0.1):
    is_terminal = False
    score = 0
    steps = 0
    start_timer = time.time()

    current_state = env.reset()

    if render:
        img = plt.imshow(env.render(mode="rgb_array"))  # only call this once

    while (not is_terminal) and (steps < max_steps):
        # get action from the agent
        action = agent.get_action(current_state)

        # advance one step in the environment
        next_state, reward, is_terminal, info = env.step(action)

        # update score
        score += reward

        # Clip the agent's reward - improves stability
        reward = np.clip(reward, -1, 1)

        if train:
            # save the transition
            agent.save_experience(current_state, action, reward, next_state, is_terminal)

        # update current state
        current_state = next_state

        if render:
            # render the frame and pause
            img.set_data(env.render(mode="rgb_array"))  # just update the data
            display.display(plt.gcf())
            display.clear_output(wait=True)
            time.sleep(sleep)

        steps += 1

    # Call to clear internal agent statistics (reward, return, etc.)
    loss = agent.finalize_episode()

    total_time = time.time() - start_timer
    return {
        "score": score,
        "steps_per_game": steps,
        "framerate": steps / (total_time + 1e-6),
        "loss": loss,
        "epsilon": agent.exploration.epsilon,
        "time_per_game": total_time,
    }


# Define loop for multiple episodes
def run_experiment(
    env,
    agent,
    runs=100,
    x_plot=None,
    plot_stats=None,
    history={},
    **kwargs,
):

    total_steps = 0
    history["total_steps"] = []

    if plot_stats:
        num_plots = len(plot_stats)
        fig, axs = plt.subplots(num_plots, 1, squeeze=False, figsize=(10, 5 * num_plots))
        axs = axs.ravel()

    for i in range(runs):
        stats = run_episode(env, agent, **kwargs)

        #       Update history object
        for k, v in stats.items():
            if k not in history.keys():
                history[k] = []
            history[k].append(v)

        total_steps += history["steps_per_game"][-1]
        history["total_steps"].append(total_steps)

        if plot_stats:
            for ax, stat_name in zip(axs, plot_stats):
                ax.clear()
                if x_plot is None:
                    sns.lineplot(
                        x=np.arange(len(history[stat_name])), y=history[stat_name], ax=ax
                    )
                else:
                    sns.lineplot(x=history[x_plot], y=history[stat_name], ax=ax)
                ax.set_title(stat_name)
            display.display(fig)
            display.clear_output(wait=True)
        else:
            ...
            print(f"episode {i}/{runs} | {stats}", )





if __name__ == "__main__":
    import gym

    import yaml

    from os.path import join




    from gym.wrappers import TransformObservation

    # config = yaml.load(open(join("configs", "cartole_ddqn.yaml")), Loader=yaml.FullLoader)
    # env = TransformObservation(gym.make("Pong-ramDeterministic-v4"),
    #                            f=lambda x: x.astype('float') / 255.)
    # env = gym.make("CartPole-v0")
    # env = gym.make("LunarLander-v2")
    from env_wrappers import StackEnvWrapper

    config = yaml.load(open(join("configs", "cnn_dqn.yaml")), Loader=yaml.FullLoader)
    env = StackEnvWrapper(
        gym.make("PongNoFrameskip-v4"),
        frame_skip=4,
        terminal_on_life_loss=False,
        screen_size=84,
        stack_size=4,
        skip_init=1,
    )

    agent = QAgent(
        observation_shape=env.observation_space.shape,
        n_actions=env.action_space.n,
        **config,
    )

    history = {}
    run_experiment(
        env,
        agent,
        train=True,
        runs=500,
        plot_stats=None,
        history=history,
    )
    plot_stats = ['score']
    x_plot=None
    num_plots = len(plot_stats)
    fig, axs = plt.subplots(num_plots, 1, squeeze=False, figsize=(10, 5 * num_plots))
    axs = axs.ravel()
    for ax, stat_name in zip(axs, plot_stats):
        ax.clear()
        if x_plot is None:
            sns.lineplot(
                x=np.arange(len(history[stat_name])), y=history[stat_name], ax=ax
            )
        else:
            sns.lineplot(x=history[x_plot], y=history[stat_name], ax=ax)
        ax.set_title(stat_name)
    plt.show()