from collections import Counter, deque
import numpy as np

import random


class EpsilonGreedy:
    def __init__(self, n_outputs, epsilon_min = 0.05, epsilon_max = 1.0, 
                 epsilon_decay=0.999):
        self.n_outputs = n_outputs
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_decay = epsilon_decay 
        self.epsilon = epsilon_max
    
    def get_action(self, action):
        self.epsilon = self.epsilon*self.epsilon_decay + (1-self.epsilon_decay)*self.epsilon_min
        if random.random() < self.epsilon:
            return np.random.randint(self.n_outputs)
        else:
            return action
    
class ExperienceReplay:
    def __init__(self, buffer_len, batch_size):
        self.exp_buffer = deque(maxlen=buffer_len)
        self.batch_size = batch_size
    
    def sample_memories(self):
        perm_batch = np.random.choice(range(len(self.exp_buffer)), size=self.batch_size, replace=False)
        mem = np.array(self.exp_buffer)[perm_batch]
        return mem[:,0], mem[:,1], mem[:,2], mem[:,3], mem[:,4]
    
    def add(self, item):
        self.exp_buffer.append(item)