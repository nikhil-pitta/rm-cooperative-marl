from reward_machines.managed_sparse_reward_machine import ManagedSparseRewardMachine
from tester.tester import Tester
from infrastructure.replay_buffer import ReplayBuffer
import infrastructure.pytorch_utils as ptu
import numpy as np
import random, time, os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from Manager.manager import Manager
import wandb
import yaml

class Agent:
    """
    Class meant to represent an individual RM-based learning agent.
    The agent maintains a representation of its own q-function and accumulated reward
    which are updated across training episodes.
    The agent also has a representation of its own local reward machine, which it uses
    for learning, and of its state in the world/reward machine.
    
    Note: Users of this class must manually reset the world state and the reward machine
    state when starting a new episode by calling self.initialize_world() and 
    self.initialize_reward_machine().
    """
    def __init__(self, rm_file, s_i, num_states, actions, agent_id, batch_size, buffer_size, tester=None, manager:Manager=None):
        """
        Initialize agent object.

        Parameters
        ----------
        rm_file : str
            File path pointing to the reward machine this agent is meant to use for learning.
        s_i : int
            Index of initial state.
        actions : list
            List of actions available to the agent.
        agent_id : int
            Index of this agent.
        """

        self.tester = tester

        self.rm_file = rm_file
        self.agent_id = agent_id
        self.s_i = s_i
        self.s = s_i
        self.actions = actions
        self.num_states = num_states
        self.curr_loss = 0

        self.rm = ManagedSparseRewardMachine(self.rm_file)
        # self.u = self.rm.get_initial_state()
        # self.local_event_set = self.rm.get_events()
        self.local_event_set = manager.get_events(self.agent_id)
        
        self.q = np.zeros([num_states, len(self.rm.U), len(self.actions)])
        self.total_local_reward = 0
        self.is_task_complete = 0
        self.batch_size = batch_size
        self.target_network_update_period = 100

        self.buffer = ReplayBuffer(buffer_size)


        self.Q = ptu.build_mlp(
            input_size=2,
            output_size = len(self.actions),
            n_layers=tester.config['num_layers'],
            size=tester.config['layer_size'],
        )
        self.Q_target = ptu.build_mlp(
            input_size=2,
            output_size = len(self.actions),
            n_layers=tester.config['num_layers'],
            size=tester.config['layer_size'],
        )
        
        self.optimizer = torch.optim.Adam([
                        {'params': self.Q.parameters(), 'lr': tester.config['learning_rate']},
                    ], betas=[tester.config['adam_betas'], tester.config['adam_betas']])

        self.update_target_network()

        self.loss = nn.SmoothL1Loss()

    def reset_state(self):
        """
        Reset the agent to the initial state of the environment.
        """
        self.s = self.s_i

    # def initialize_reward_machine(self):
    #     """
    #     Reset the state of the reward machine to the initial state and reset task status.
    #     """
    #     self.u = self.rm.get_initial_state()
    #     self.is_task_complete = 0

    def is_local_event_available(self, label):
        if label: # Only try accessing the first event in label if it exists
            event = label[0]
            return self.rm.is_event_available(self.u, event)
        else:
            return False

    def get_next_action(self, epsilon, learning_params):
        """
        Return the action next action selected by the agent.

        Outputs
        -------
        s : int
            Index of the agent's current state.
        a : int
            Selected next action for this agent.
        """

        curr_state = np.row_stack(([self.s], [self.u])).T

        # T = learning_params.T

        if np.random.rand() < epsilon:
            a = np.random.choice(self.actions)
        else:
            qa = self.Q(ptu.from_numpy(curr_state).float())
            a = torch.argmax(qa)
            a = ptu.to_numpy(a).squeeze(0).item()
        
        return self.s, a

    def update_agent(self, s_new, reward, label, learning_params, step, update_q_function=True, i=-1, evaluate_critic_loss=False):
        """
        Update the agent's state, q-function, and reward machine after 
        interacting with the environment.

        Parameters
        ----------
        s_new : int
            Index of the agent's next state.
        a : int
            Action the agent took from the last state.
        reward : float
            Reward the agent achieved during this step.
        label : string
            Label returned by the MDP this step.
        learning_params : LearningParameters object
            Object storing parameters to be used in learning.
        """

        # # Keep track of the RM location at the start of the 
        # u_start = self.u

        for event in label: # there really should only be one event in the label provided to an individual agent
            # Update the agent's RM
            u2 = self.rm.get_next_state(self.u, event)
            self.u = u2
        
        self.total_local_reward += reward
        # print("reward", self.rm.is_terminal_state(self.u), "isDone", self.rm.is_terminal_state(self.u))
        # if i >= 0:
        #     wandb.log({f"Reward Achieved for Agent {i}": int(reward), "Step": self.tester.get_global_step()})
        if update_q_function == True and step < self.tester.early_stopping_point:
            self.curr_loss = self.update_q_function(learning_params, step)
        
        # if evaluate_critic_loss:
        #      = self.eval_q_loss(self.s, s_new, u_start, self.u, a, reward, learning_params, step)
        # # else:
        #     wandb.log({f"Critic Loss for Agent {i}": 1, 'Step': self.tester.get_global_step()})

        # Moving to the next state
        self.s = s_new

        if self.rm.is_terminal_state(self.u):
            # Completed task. Set flag.
            self.is_task_complete = 1


    def update_q_function(self, learning_params, step):
        """
        Update the q function using the action, states, and reward value.

        Parameters
        ----------
        s : int
            Index of the agent's previous state
        s_new : int
            Index of the agent's updated state
        u : int
            Index of the agent's previous RM state
        U_new : int
            Index of the agent's updated RM state
        a : int
            Action the agent took from state s
        reward : float
            Reward the agent achieved during this step
        learning_params : LearningParameters object
            Object storing parameters to be used in learning.
        """
        alpha = learning_params.alpha
        gamma = learning_params.gamma

        # Bellman update
        # self.q[s][u][a] = (1-alpha)*self.q[s][u][a] + alpha*(reward + gamma*np.amax(self.q[s_new][u_new]))
        #Q-function update
        states, rm_states, actions, rewards, next_states, next_rm_states = self.buffer.sample(self.batch_size)
        states, rm_states, actions, rewards, next_states, next_rm_states = map(ptu.from_numpy, [np.array(states), np.array(rm_states), np.array(actions), np.array(rewards), np.array(next_states), np.array(next_rm_states)])
        next_input = ptu.from_numpy(np.row_stack((np.array(next_states), np.array(next_rm_states))).T).float()
        curr_input = ptu.from_numpy(np.row_stack((np.array(states), np.array(rm_states))).T).float()
        

        next_qa_values = self.Q_target(next_input)

        next_action = torch.argmax(self.Q(next_input), dim=1, keepdim=True)

        next_q_values = torch.gather(next_qa_values, 1, next_action).squeeze()

        target_values = rewards + (gamma * (next_q_values * (1-  ptu.from_numpy(np.array([self.rm.is_terminal_state(i) for i in rm_states])).float())))
        
        qa_values = self.Q(curr_input)
        q_values = torch.gather(qa_values, 1, torch.unsqueeze(actions, dim=-1)).squeeze()
        # Compute from the data actions; see torch.gather
        loss = self.loss(q_values, target_values)

        # print("i", i)
        # if i >= 0:
        #     print("broooo")
        #     wandb.log({f"Critic Loss for Agent {i}": loss, 'Step': self.tester.get_global_step()})


        self.optimizer.zero_grad()
        loss.backward()
        # grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
        #     self.Q.parameters(), self.clip_grad_norm or float("inf")
        # )
        self.optimizer.step()

        if step % self.target_network_update_period == 0:
            self.update_target_network()

        return loss
        
    def update_target_network(self):
        # copy current_network to target network
        self.Q_target.load_state_dict(self.Q.state_dict())