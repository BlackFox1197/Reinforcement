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
num_test_episodes = 1

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

#
#
# num_episodes = 50000        # Total episodes
# max_steps_per_episode = 100 # Max steps per episode
# num_test_episodes = 5     # Total test episodes
#
# rewards_of_all_episodes = []
#
#
# environment_data_object = env #Environment(action_space, observation_space)
# environment = enviorm.EnvironmentRepositoryImpl(environment_data_object)
#
# # Setting up the Agent
# agent_data_object = agent.Agent()
# agent = agent.AgentRepositoryImpl(environment_data_object, agent_data_object, HIDDEN_NEURONS)
#
# n_bins = ( 6 , 12 )
# lower_bounds = [ env.observation_space.low[2], -math.radians(50) ]
# upper_bounds = [ env.observation_space.high[2], math.radians(50) ]
#
# def discretizer( _ , __ , angle, pole_velocity ) -> Tuple[int,...]:
#     """Convert continues state intro a discrete state"""
#     est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
#     est.fit([lower_bounds, upper_bounds ])
#     return tuple(map(int,est.transform([[angle, pole_velocity]])[0]))
#
#
# # Training Loop
# for episode in range(num_episodes):
#     # reset/initialize the environment first
#     current_state = discretizer(*env.reset())
#     #state = environment.reset()
#     state = discretizer(*env.reset())
#     # set done back to false at the beginning of an episode
#     done = False
#     # reset our rewards collector | return for the beginning episode
#     rewards_current_episode = 0
#
#     for step in range(max_steps_per_episode):
#         # select an action
#         # use our exploration exploitation trade off -> do we explore or exploit in this timestep ?
#         action = agent.get_action(state)
#
#         # increment enviroment
#         new_state, obs, reward, done, _ = env.step(action)
#         new_state = discretizer(*new_state)
#
#         #new_state, reward, done, info = environment.step(action)
#
#
#         # Update Q-Table Q(s,a) using the bellman update
#         agent.update_q_table(state, action, reward, new_state)
#
#         # update the state to the new state
#         state = new_state
#         # collect the reward
#         rewards_current_episode += reward
#
#         if (done == True):
#             break
#
#     # after we finish an episode, make sure to update the exploration rate
#     # decay the exploration rate the longer the time goes on
#     agent.update_exploration_rate(episode)
#     # append our rewards for this episode for learning curve
#     rewards_of_all_episodes.append(rewards_current_episode)
#
#
#
#
#
#     # LEARNING STATISTICS
#     # for each episode print the stats of the episode
#     rewards_per_thousand_episodes = np.split(np.array(rewards_of_all_episodes), num_episodes / 1000)
#     count = 1000
#     print('*****INFO: average reward per thousand episodes: ***** \n')
#     for reward in rewards_per_thousand_episodes:
#         print(count, ": ", str(sum(reward / 1000)))
#         count += 1000
#
#     # print our learned q-table
#     print("\n\n ***** Q-TABLE ***** \n")
#     print(agent.q_table)
#
#
#
#
#     # EVALUATION | TESTING | watching our agent play
#     for episode in range(num_test_episodes):
#         state = environment.reset()
#         done = False
#         print("INFO:*****EPISODE ", episode + 1, "\n\n\n")
#         time.sleep(1)
#
#         for step in range(max_steps_per_episode):
#             clear_output(wait=True)
#             environment.render()
#             time.sleep(0.3)
#
#             action = agent.get_exploit_action(state)
#             new_state, reward, done, info = environment.step(action)
#
#             if done:
#                 clear_output(wait=True)
#                 environment.render()
#                 if reward == 20:
#                     print("INFO: ***** agent reached the goal. *****")
#                     time.sleep(3)
#                 else:
#                     print("INFO: ***** agent missed the goal.")
#                     time.sleep(3)
#                 clear_output(wait=True)
#                 break
#
#             state = new_state
#
#     environment.close()