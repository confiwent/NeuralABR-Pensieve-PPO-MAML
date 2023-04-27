import numpy as np
import random
import torch

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, events):
        for event in zip(*events):
            self.memory.append(event) # event shape (state, action, ...)
            if len(self.memory) > self.capacity:
                del self.memory[0]

    def clear(self):
        self.memory = []

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size)) # samples shape: [(states), (actions)]
        # samples = zip(*self.memory[:batch_size])
        #return map(lambda x: torch.cat(x, 0), samples)
        return map(lambda x: np.array(x), samples)

    def sample_cuda(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size)) # samples shape: [(states), (actions)]
        return map(lambda x: torch.cat(x, 0), samples)

    def pop(self, batch_size):
        mini_batch = zip(*self.memory[:batch_size])
        #return map(lambda x: torch.cat(x, 0), mini_batch)
        return map(lambda x: np.array(x), mini_batch)

    def return_size(self):
        return len(self.memory)

# lambd x is an anonymous function 
# = def (x):
# class ReplayMemory(object):
#     def __init__(self):
#         self.capacity = 0
#         self.memory = []
#         self.all_truples = []
#         self.priority = {}

#     def push(self, events):
#         for event in zip(*events):
#             action = event[1]
#             reward = event[2]
#             if (action, reward) not in self.all_truples:
#                 self.memory.append(event)
#                 self.capacity += 1
#                 self.priority[(action, reward)] = 1
#             else:
#                 existing_num = self.priority[(action, reward)]
#                 if existing_num/self.capacity <= 0.5 or self.capacity <= MIN_MOMERY: 
#                     self.memory.append(event)
#                     self.capacity += 1
#                     self.priority[(action, reward)] += 1
#             # self.memory.append(event)
#             # if len(self.memory) > self.capacity:
#             #     del self.memory[0]

#     def clear(self):
#         self.memory = []
#         self.capacity = 0

#     def get_capacity(self):
#         return self.capacity

#     def sample(self, batch_size):
#         samples = zip(*random.sample(self.memory, batch_size))
#         return map(lambda x: torch.cat(x, 0), samples)
