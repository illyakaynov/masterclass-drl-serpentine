from dqn.agent import QAgent
import numpy as np
import gym
from IPython import display
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
import time

def run_episode(env, agent, train=False, max_steps=100_000, render=False, sleep=0.1):
    is_terminal = False
    score = 0
    steps = 0
    start_timer = time.time()
    current_state = env.reset()

    if render:
        img = plt.imshow(env.render(mode='rgb_array'))  # only call this once

    while (not is_terminal) and (steps < max_steps):
        # get action from the agent
        action = agent.get_action(current_state)

        # advance one step in the environment
        next_state, reward, is_terminal, info = env.step(action)

        # update score
        score += reward

        reward = np.clip(reward, -1, 1)
        # save the transition
        if train:
            agent.update(current_state, action, reward, next_state, is_terminal)
        # update current state
        current_state = next_state

        if render:
            img.set_data(env.render(mode='rgb_array'))  # just update the data
            display.display(plt.gcf())
            display.clear_output(wait=True)
            time.sleep(sleep)
    agent.finalize_episode()

    total_time = time.time() - start_timer
    return {'score': score, 'steps': steps, 'framerate': steps / total_time}