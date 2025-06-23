import numpy as np
import random

class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.nA = n_actions
        self.nS = n_states
        # Q-table where each entry estimates expected reward of taking action a in state s
        self.Q = np.zeros((n_states, n_actions))
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.nA)
        return int(np.argmax(self.Q[state]))

    def update(self, s, a, r, s_next, done):
        # finds highest Q-value among all possible actions in next state s_next
        best_next = np.max(self.Q[s_next])
        # if episode over no future reward beyond r so target is just r, otherwise bootstrapp off the best next-state value, discounted by gamma 
        target = r + (0 if done else self.gamma * best_next)
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])