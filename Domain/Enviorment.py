from abc import ABC, abstractmethod
import numpy as np
import random

# DOMAIN = the data_object (JSON SERIALIZABLE)
class Environment():
    """The main Environment class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.
    The main API methods that users of this class need to know are:
        step
        reset
        render
        close
        seed
    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
    The methods are accessed publicly as "step", "reset", etc...
    """
    # Set this in SOME subclasses
    metadata = {'render.modes': []}
    reward_range = (-float('inf'), float('inf'))
    spec = None

    def __init__(self, action_space=None, observation_space=None):
        # Set variables
        self.action_space = action_space
        self.observation_space = observation_space


# REPOSITORY = the functionality interface
class EnvironmentRepository(ABC):
    @abstractmethod
    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        pass

    @abstractmethod
    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.
        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.
        Returns:
            observation (object): the initial observation.
        """
        pass

    @abstractmethod
    def render(self, mode='human'):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    # just raise an exception
                    super(MyEnv, self).render(mode=mode)
        """
        pass

    @abstractmethod
    def close(self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass


# REPOSITORY IMPLEMENTATION = the way how you would like to implement it
class EnvironmentRepositoryImpl(EnvironmentRepository):
    # Initialize / Instance Attributes
    def __init__(self, environment):
        # Set variables
        self.data_object = environment
        print('Environment initialized')

    def step(self, action):
        state = self.data_object.step(action)
        return state

    def reset(self):
        state = self.data_object.reset()
        return state

    def render(self):
        self.data_object.render()

    def close(self):
        state = self.data_object.close()

    def get_action_space(self):
        # get action space from api of the playground or via js in browser using selenium
        action_space = self.data_object.action_space
        return action_space

    def get_observation_space(self):
        # get observation space of the playground from api or via js in browser using selenium
        observation_space = self.data_object.observation_space
        return observation_space
