import numpy as np

class ReplayBuffer:
    def __init__(self, mem_size, state_shape):
        self.mem_size = mem_size
        self.mem_count = 0

        self.states       = np.zeros((self.mem_size, *state_shape),dtype=np.float32)
        self.actions      = np.zeros( self.mem_size,               dtype=np.int64  )
        self.rewards      = np.zeros( self.mem_size,               dtype=np.float32)
        self.states_      = np.zeros((self.mem_size, *state_shape),dtype=np.float32)
        self.dones        = np.zeros( self.mem_size,               dtype=np.bool   )
        self.invalid_move = np.zeros( self.mem_size,               dtype=np.bool   )

    def add(self, state, action, reward, state_, done, invalid_move):
        mem_index = self.mem_count % self.mem_size 
        
        self.states[mem_index]       = state
        self.actions[mem_index]      = action
        self.rewards[mem_index]      = reward
        self.states_[mem_index]      = state_
        self.dones[mem_index]        = done
        self.invalid_move[mem_index] = invalid_move

        self.mem_count += 1

    def fetch_samples(self, indices):
        states       = self.states[indices]
        actions      = self.actions[indices]
        rewards      = self.rewards[indices]
        states_      = self.states_[indices]
        dones        = self.dones[indices]
        invalid_move = self.invalid_move[indices]

        return states, actions, rewards, states_, dones, invalid_move

    def sample(self, batch_size):
        mem_max = min(self.mem_count, self.mem_size)
        batch_indices = np.random.choice(mem_max, batch_size, replace=True)
        batch = self.fetch_samples(batch_indices)
        return batch