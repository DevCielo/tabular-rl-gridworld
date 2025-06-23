import numpy as np
import random

class SarsaAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.nA = n_actions
        self.Q = np.zeros((n_states, n_actions))
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.nA)
        return int(np.argmax(self.Q[state]))

    def update(self, s, a, r, s_next, a_next, done):
        target = r + (0 if done else self.gamma * self.Q[s_next, a_next])
        self.Q[s,a] += self.alpha * (target - self.Q[s, a])