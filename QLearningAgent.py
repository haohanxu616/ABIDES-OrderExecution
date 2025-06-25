import numpy as np
from typing import Dict, List, Tuple
import logging
import gym

from Utils import logger

class QLearningAgent:
    """A Q-learning agent for algorithmic trading strategies.

    Parameters
    ----------
    action_space : gym.spaces.Discrete
        The action space of the environment.
    strategy : str
        The trading strategy (TWAP, VWAP, IS, or POV).

    Attributes
    ----------
    q_table : Dict[Tuple, float]
        Q-value table for state-action pairs.
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    epsilon : float
        Exploration rate.
    """

    def __init__(self, action_space: gym.spaces.Discrete, strategy: str):
        self.action_space = action_space
        self.strategy = strategy
        self.q_table = {}
        self.alpha = 0.1  # Initial learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.9  # Initial exploration rate
        self.alpha_decay = 0.995  # Decay rate for learning rate
        self.min_alpha = 0.01  # Minimum learning rate
        self.epsilon_decay = 0.995  # Decay rate for epsilon
        self.min_epsilon = 0.1  # Minimum exploration rate

    def get_state_hash(self, state: np.ndarray) -> Tuple:
        """Convert state array to a hashable tuple."""
        return tuple(state.flatten().astype(int))

    def choose_action(self, state: np.ndarray, env) -> int:
        """Choose an action using epsilon-greedy policy."""
        state_hash = self.get_state_hash(state)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)  # Decay epsilon
        if np.random.random() < self.epsilon:
            return env.action_space.sample()  # Explore
        else:
            if state_hash not in self.q_table:
                self.q_table[state_hash] = np.zeros(env.action_space.n)
            return np.argmax(self.q_table[state_hash])  # Exploit

    def learn(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, env) -> None:
        """Update Q-values based on the Q-learning update rule."""
        state_hash = self.get_state_hash(state)
        next_state_hash = self.get_state_hash(next_state)
        if state_hash not in self.q_table:
            self.q_table[state_hash] = np.zeros(env.action_space.n)
        if next_state_hash not in self.q_table:
            self.q_table[next_state_hash] = np.zeros(env.action_space.n)
        current_q = self.q_table[state_hash][action]
        next_max_q = np.max(self.q_table[next_state_hash])
        self.alpha = max(self.min_alpha, self.alpha * self.alpha_decay)  # Decay learning rate
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state_hash][action] = new_q