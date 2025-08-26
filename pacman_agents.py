# pacman_agents.py
import random
import numpy as np

class BaseAgent:
    def choose_action(self, observation):
        raise NotImplementedError("This method should be overridden by subclasses")

class RandomAgent(BaseAgent):
    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self, observation):
        return self.action_space.sample()

class RuleBasedAgent(BaseAgent):
    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self, frame):
        mask = (frame[:, :, 0] > 150) & (frame[:, :, 1] < 100) & (frame[:, :, 2] < 100)
        if np.any(mask):
            return self.action_space.sample()
        else:
            return 3  

class ExpectimaxAgent(BaseAgent):
    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self, frame):
        scores = []
        for action in range(self.action_space.n):
            score = self.evaluate_action(frame, action)
            scores.append(score)
        return int(np.argmax(scores))

    def evaluate_action(self, frame, action):
        mask = (frame[:, :, 0] > 150) & (frame[:, :, 1] < 100) & (frame[:, :, 2] < 100)
        if np.any(mask):
            return -10
        else:
            return random.randint(0, 10)

class QLearningAgent(BaseAgent):
    def __init__(self, action_space, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.action_space = action_space
        self.q_table = dict()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def preprocess(self, frame):
        gray = np.mean(frame, axis=2)
        small = gray[::7, ::7]
        return tuple((small > 100).flatten())

    def choose_action(self, frame):
        state = self.preprocess(frame)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space.n)
        if random.random() < self.epsilon:
            return self.action_space.sample()
        return int(np.argmax(self.q_table[state]))

    def update(self, frame, action, reward, next_frame):
        state = self.preprocess(frame)
        next_state = self.preprocess(next_frame)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_space.n)
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action]
        )
