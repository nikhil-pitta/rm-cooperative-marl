import numpy as np
import pandas as pd
class ReplayBuffer:
    def __init__(self, max_ = 10000) -> None:
        self.max_ = max_
        self.counter = -1

        self.states = np.array([0 for _ in range(max_)])
        self.actions = np.array([0 for _ in range(max_)])
        self.rewards = np.array([0 for _ in range(max_)])
        self.next_states = np.array([0 for _ in range(max_)])
        self.rm_states = np.array([0 for _ in range(max_)])
        self.next_rm_states = np.array([0 for _ in range(max_)])

        self.current_traj = []
        self.all_current_traj = []
    
    def add(self, s, u, a, r, s_new, u2):
        self.counter += 1
        self.states[self.counter % self.max_] = s
        self.rm_states[self.counter % self.max_] = u
        self.next_states[self.counter % self.max_] = s_new
        self.next_rm_states[self.counter % self.max_] = u2
        self.actions[self.counter % self.max_] = a
        self.rewards[self.counter % self.max_] = r
        self.all_current_traj.append(self.counter % self.max_)
    
    def mark(self):
        self.current_traj.append(self.counter % self.max_)
    
    def restart_traj(self):
        self.current_traj = []
        self.all_current_traj = []
    
    # def get_current_traj(self):
    #     idxs = np.array(self.current_traj)
    #     df = pd.DataFrame([self.states[idxs], self.rm_states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], self.next_rm_states[idxs]]).T
    #     df.columns = ['s', 'b', 'a', 'r', 's_', 'b_']

    #     idxs = np.array(self.all_current_traj)
    #     df2 = pd.DataFrame([self.states[idxs], self.rm_states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], self.next_rm_states[idxs]]).T
    #     df2.columns = ['s', 'b', 'a', 'r', 's_', 'b_']
    #     return df, df2
    
    def get_all(self):
        idxs = np.arange(0, min(self.counter, self.max_-1))
        return self.states[idxs], self.rm_states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], self.next_rm_states[idxs]

    def sample(self, batchsize):
        idxs = np.random.random_integers(0, min(self.counter, self.max_-1), batchsize)
        return self.states[idxs], self.rm_states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], self.next_rm_states[idxs]