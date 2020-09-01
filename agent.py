import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from model import Network
from replay_buffer import ReplayBuffer

class Lerper:
    def __init__(self, start, end, num_steps):
        self.delta = (end - start) / float(num_steps)
        self.num = start - self.delta
        self.count = 0
        self.num_steps = num_steps

    def value(self):
        return self.num

    def step(self):
        if self.count <= self.num_steps:
            self.num += self.delta
        self.count += 1
        return self.num

class Agent:
    def __init__(self, lr, state_shape, num_actions, batch_size, 
            max_mem_size=100000):
        self.lr = lr
        self.gamma = 0.99
        self.action_space = list(range(num_actions))
        self.batch_size = batch_size

        self.epsilon = Lerper(start=1.0, end=0.01, num_steps=2000)

        self.memory = ReplayBuffer(max_mem_size, state_shape)
        self.net = Network(lr, inputChannels = 3, numActions = 9)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon.value():
            state = torch.tensor(observation).float().detach()
            state = state.to(self.net.device)
            state = state.unsqueeze(0)

            q_values = self.net(state)
            action = torch.argmax(q_values).item()
            return action
        else:
            return np.random.choice(self.action_space)

    def store_memory(self, state, action, reward, state_, done, 
            invalid_move):
        self.memory.add(state, action, reward, state_, done, invalid_move)

    def learn(self):
        if self.memory.mem_count < self.batch_size:
            return

        states, actions, rewards, states_, dones, invalid_moves = \
            self.memory.sample(self.batch_size)
        states       = torch.tensor( states        ).to(self.net.device)
        actions      = torch.tensor( actions       ).to(self.net.device)
        rewards      = torch.tensor( rewards       ).to(self.net.device)
        states_      = torch.tensor( states_       ).to(self.net.device)
        dones        = torch.tensor( dones         ).to(self.net.device)
        invalid_move = torch.tensor( invalid_moves ).to(self.net.device)

        batch_index = np.arange(self.batch_size, dtype=np.int64)

        q_values  =   self.net(states)[batch_index, actions]
        q_values_ =   self.net(states_)

        action_qs_ = torch.max(q_values_, dim=1)[0]
        action_qs_[dones] = 0.0
        q_target = rewards + self.gamma * action_qs_

        td = q_target - q_values

        self.net.optimizer.zero_grad()
        loss = (td ** 2.0).mean()
        loss.backward()
        self.net.optimizer.step()

        self.epsilon.step()

if __name__ == "__main__":
    state_shape = (3,3,3)

    #   make agent
    agent = Agent(lr=0.001, state_shape=state_shape, num_actions=9, batch_size=64)

    #   make fake data
    x = np.ones((1, *state_shape))
    print("Fake data {}".format(x.shape))

    #   compute some actions
    action = agent.choose_action(x)
    print("action {}".format(action))

    #   fill memory
    for i in range(agent.batch_size):
        agent.store_memory(
            state=x, action=0, reward=0, state_=x, done=False, 
            invalid_move=False)


    #   test learn
    agent.learn()

    print("TEST DONE")

