from time import time

import numpy as np
np.random.seed(0)


class FlappyAgent:

    def __init__(self, observation_space_size, action_space, n_iterations):
        self.q_table = np.zeros([*observation_space_size, len(action_space)])
        self.env_action_space = action_space
        self.n_iterations = n_iterations

        self.alpha = 0.5
        self.gamma = 0.5
        self.epsilon = 0.13
        self.decay = 0.02

        self.test = False

    def step(self, state):
        action = 0

        if not self.test and np.random.uniform() > self.epsilon:
            maxval = max(self.q_table[state])
            action = list(self.q_table[state]).index(maxval)
        else:
            action = np.argmax(self.q_table[state])
            self.epsilon -= self.decay * self.epsilon

        return action

    def epoch_end(self, epoch_reward_sum):
        pass

    def learn(self, old_state, action, new_state, reward):
        q_val = self.q_table[old_state + (action,)]
        maxval = max([self.q_table[new_state + (act,)]
                      for act in self.env_action_space])
        self.q_table[old_state + (action,)] += self.alpha * \
            (reward + self.gamma * maxval - q_val)

    def train_end(self):
        
        #self.q_table = None  # TODO
        self.test = True
