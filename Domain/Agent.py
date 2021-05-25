from abc import ABC, abstractmethod
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F

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
        self.observation_space = self.environment.observation_space
        self.action_space_size = self.environment.action_space.n
        #self.observation_space_size = self.environment.observation_space.n
        self.observation_space_size = (6, 12)
        # INIT Q-TABLE
        self.q_table = np.zeros((self.observation_space_size, self.action_space_size))
        self.q_table = Network(hidden_neurons)
        # INIT AGENT PARAMETERS
        self.learning_rate = 0.7  # Learning rate
        self.discount_rate = 0.618  # Discounting rate
        self.exploration_rate = 1.0  # Exploration rate
        self.max_exploration_rate = 1.0  # Exploration probability at start
        self.min_exploration_rate = 0.01  # Minimum exploration probability
        self.exploration_decay_rate = 0.01  # Exponential decay rate for exploration probability
        print('Agent initialized.')

    def get_action(self, state):
        # EXPLORATION-EXPLOITATION TRADE OFF
        exploration_rate_threshold = random.uniform(0, 1)
        if (exploration_rate_threshold > self.exploration_rate):
            # get action from q table
            action = np.argmax(self.q_table[state, :])
        else:
            # get random action
            action = self.get_random_action()
        return action

    def get_random_action(self):
        # action_set = random.sample(self.action_space, 1)
        # action = action_set[0]
        action = self.action_space.sample()
        return action

    def update_q_table(self, state, action, reward, new_state):

        self.q_table[state, action] = self.q_table[state, action] * (1 - self.learning_rate) + self.learning_rate * (
                    reward + self.discount_rate * np.max(self.q_table[new_state, :]))

    def update_exploration_rate(self, episode_num):
        self.exploration_rate = self.min_exploration_rate + (
                    self.max_exploration_rate - self.min_exploration_rate) * np.exp(
            -self.exploration_decay_rate * episode_num)

    def get_exploit_action(self, state):
        action = np.argmax(self.q_table[state, :])
        return action



class Network(nn.Module):
    def __init__(self, hidden_neurons):
        self.l1 = nn.Linear(4, hidden_neurons)
        self.l2 = nn.Linear(hidden_neurons, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x
