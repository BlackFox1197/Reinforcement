from abc import ABC, abstractmethod
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import torch

from Domain.neural_network import DeepQNetwork


class Agent():
    # class variables
    agent_variable = ""

    # class methods
    def __init__(self, agent_variable=""):
        self.agent_variable = agent_variable


# REPOSITORY
class AgentRepository(ABC):
    @abstractmethod
    def get_action(self, state):  # This is the agent's POLICY, if you will
        """ Agent gets a state as input and returns an action
        """
        pass


class AgentRepositoryImpl(AgentRepository):
    # Initializer / Instance Attributes
    def __init__(self, environment, agent, hidden_neurons):
        # Set variables
        self.data_object = agent
        self.environment = environment
        self.action_space = self.environment.action_space
        self.action_space_size = self.environment.action_space.n

        self.observation_space = self.environment.observation_space
        #self.observation_space_size = self.environment.observation_space.n
        self.observation_space_size = environment.observation_space.shape[0]
        self.state_space_size = environment.observation_space.shape[0]
        # INIT Q-TABLE
        #self.q_table = Network(hidden_neurons)
        # INIT AGENT PARAMETERS
        self.learning_rate = 0.7  # Learning rate
        self.discount_rate = 0.618  # Discounting rate
        self.exploration_rate = 1.0  # Exploration rate
        self.max_exploration_rate = 1.0  # Exploration probability at start
        self.min_exploration_rate = 0.01  # Minimum exploration probability
        self.exploration_decay_rate = 0.01  # Exponential decay rate for exploration probability

        self.deep_q_network = DeepQNetwork(self.state_space_size, self.action_space_size, hidden_neurons, self.learning_rate)
        print('Agent initialized.')

    def get_action(self, state):
        exploration_rate_threshold = random.uniform(0,1)
        if(exploration_rate_threshold > self.exploration_rate):
            q_values = self.deep_q_network.predict(state)
            action = torch.argmax(q_values).item()
        else:
            action = self.environment.action_space.sample()
        return action

    # def get_random_action(self):
    #
    #     # action_set = random.sample(self.action_space, 1)
    #     # action = action_set[0]
    #     action = self.action_space.sample()
    #     return action

    def update_q_table(self, state, action, reward, new_state, done):

        # Update our Q Network due to the reward we got
        q_values = self.deep_q_network.predict(state).tolist()
        # Update network weights using the last step only
        q_values_next = self.deep_q_network.predict(new_state)
        q_values[action] = reward + self.discount_rate * torch.max(q_values_next).item()
        self.deep_q_network.update(state, q_values)
        if (done == True):
            q_values[action] = reward
            # Update network weights
            self.deep_q_network.update(state, q_values)
            return True
        return False


    def update_exploration_rate(self, episode_num):
        self.exploration_rate = self.min_exploration_rate + (
                    self.max_exploration_rate - self.min_exploration_rate) * np.exp(
            -self.exploration_decay_rate * episode_num)

    # def get_exploit_action(self, state):
    #     action = np.argmax(self.q_table[state, :])
    #     return action


#
# class Network(nn.Module):
#     def __init__(self, hidden_neurons):
#         self.l1 = nn.Linear(4, hidden_neurons)
#         self.l2 = nn.Linear(hidden_neurons, 2)
#
#     def forward(self, x):
#         x = F.relu(self.l1(x))
#         x = self.l2(x)
#         return x
