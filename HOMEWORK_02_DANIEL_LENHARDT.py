import math
from typing import Tuple

import numpy as np
import gym
import random
import time
import torch
from sklearn.preprocessing import KBinsDiscretizer

import Domain.Agent as Agent
import Domain.Enviorment as enviorm

from IPython.display import clear_output

from Domain.neural_network import DeepQNetwork

print(torch.__version__)
print(torch.cuda.is_available())

# get an instance of the frozen lake gym environment
env = gym.make("CartPole-v1")
# get some information from the environment
action_space = env.action_space
print(action_space)
action_space_size = env.action_space.n
print(action_space_size)
state_space = env.observation_space
print(state_space)

HIDDEN_NEURONS = 64

num_episodes = 1000
max_steps_per_episode = 100
num_test_episodes = 6

agent_data_object = Agent.Agent()
agent = Agent.AgentRepositoryImpl(env, agent_data_object, HIDDEN_NEURONS)


rewards_of_all_episodes = []

# Training Loop
for episode in range(num_episodes):
    # reset/initialize the environment first
    state = env.reset()
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):
        action = agent.get_action(state)
        new_state, reward, done, info = env.step(action)
        if agent.update_q_table(state, action, reward, new_state, done):
            break

        state = new_state
        # collect the reward
        rewards_current_episode += reward
    agent.update_exploration_rate(episode)
    rewards_of_all_episodes.append(rewards_current_episode)


rewards_per_hundred_episodes = np.split(np.array(rewards_of_all_episodes),num_episodes/100)
count = 100
print('*****INFO: average reward per thousand episodes: ***** \n')
for reward in rewards_per_hundred_episodes:
    print(count, ": ", str(sum(reward/100)))
    count += 100

# EVALUATION | TESTING | watching our agent play
for episode in range(num_test_episodes):
    state = env.reset()
    done = False
    print("INFO:*****EPISODE ", episode + 1, "\n\n\n")
    time.sleep(1)

    for step in range(max_steps_per_episode):
        clear_output(wait=True)
        env.render()
        time.sleep(0.1)

        q_values = agent.deep_q_network.predict(state)
        action = torch.argmax(q_values).item()
        new_state, reward, done, info = env.step(action)

        if done:
            clear_output(wait=True)
            env.render()
            if reward == 1:  # check reward from environment for correct display
                print("INFO: ***** agent reached the goal. *****")
                time.sleep(3)
            else:
                print("INFO: ***** agent did not reach the goal.")
                time.sleep(3)
            clear_output(wait=True)
            break

        state = new_state

env.close()
