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
        steps += 1
    agent.finalize_episode()

    total_time = time.time() - start_timer
    return {'score': score, 'steps': steps, 'framerate': steps / total_time}


import gym


# Initialize configs

# Network config
network_config = dict(
    double_q=False,
    copy_steps=50,
    dueling_architecture=False,
    noisy=False,
    gradient_clipping=False,
    gradient_clipping_norm=10.0,
    learning_rate=0.001,
    gamma=0.99,
    loss='mse',
    optimizer='adam',
    mlp_n_hidden=[32, 32],
    mlp_act_f='relu',
    # mlp_initializer=tf.keras.initializers.RandomNormal(0.5),
    mlp_initializer='glorot_normal',
    mlp_value_n_hidden=64,
    mlp_value_act_f='tanh',
    # mlp_value_initializer=tf.keras.initializers.RandomNormal(0.5),
    mlp_value_initializer='glorot_normal',
    input_is_image = True,
)

# Exploration config
exploration_config = dict(
    epsilon_init=1.0,
    epsilon_min=0.02,
    epsilon_decay=0.99
)

# Replay Buffer config
replay_buffer_config = dict(
    stack_size=1,
    replay_capacity=10000,
    add_last_samples=3,
    gamma=0.99,
    observation_dtype=np.uint8,
    teminal_shape=(),
    terminal_dtype=np.bool,
    action_shape=(),
    action_dtype=np.int32,
    reward_shape=(),
    reward_dtype=np.float32)

from env_wrappers import StackEnvWrapper
env = StackEnvWrapper(gym.make('PongNoFrameskip-v4'),
                 frame_skip=4,
                 terminal_on_life_loss=False,
                 screen_size=84,
                 stack_size=4,
                 skip_init=1)

agent = QAgent(observation_shape=env.observation_space.shape,
               n_actions=env.action_space.n,
               gamma=0.99, # discount of future rewards
               training_start=50, # start training after x number of steps
               training_interval=1, # train every x steps
               batch_size=32,
               priority_replay=True,
               network_params=network_config,
               exploration_params=exploration_config)

history = run_episode(env, agent, train=True)
print(history)