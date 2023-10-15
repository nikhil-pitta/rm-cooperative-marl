import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
import numpy.ma as ma
import pandas as pd

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


class DQN(nn.Module):
    def __init__(self, env_space, act_space, param):
        super(DQN, self).__init__()
        
        self.actor = nn.Sequential(
                        nn.Linear(env_space['mdp'].shape[0], 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, env_space['buchi'].n * act_space['total'])
                    )
        self.shp = (env_space['buchi'].n, act_space['total'])
            
        # Mask actions not available
        self.mask = torch.ones((env_space['buchi'].n, act_space['total'])).type(torch.bool)
        #import pdb; pdb.set_trace()
        if act_space['total'] != act_space['mdp'].n:
            for buchi in range(env_space['buchi'].n):
                try:
                    eps = act_space['total'] - 1 + act_space[buchi].n
                except:
                    eps = act_space['total'] - 1
                self.mask[buchi, eps:] = False

        self.gamma = param['gamma']
        self.n_mdp_actions = act_space['mdp'].n
        
    def forward(self, state, buchi, to_mask=True):
        all_qs = torch.reshape(self.actor(state), (-1,) + self.shp)
        qs = torch.take_along_dim(all_qs, buchi, dim=1).squeeze()
        if to_mask:
            out = torch.masked.MaskedTensor(qs, self.mask[buchi].squeeze())
        else:
            out = qs
        return out
    
    def act(self, state, buchi_state):
        qs = torch.reshape(self.actor(state), self.shp)[buchi_state]
        masked_qs = torch.masked.MaskedTensor(qs, self.mask[buchi_state])
        act = int(masked_qs.argmax())
        is_eps = act >= self.n_mdp_actions
        return act, is_eps, 0
    
    def random_act(self, state, buchi_state):
        X = self.mask[buchi_state].numpy()
        pos = np.random.choice(sum(X), size=1)
        idx = np.take(X.nonzero(), pos, axis=1)
        act = idx[0][0]
        is_eps = act >= self.n_mdp_actions
        return act, is_eps, 0
        