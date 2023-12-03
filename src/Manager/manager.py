from reward_machines.managed_sparse_reward_machine import ManagedSparseRewardMachine
import numpy as np
import infrastructure.pytorch_utils as ptu
import torch
import random
import itertools
class Manager:
    def __init__(self, rm_file, set_list, event_list, num_agents, assignment_method = "ground_truth"):
        self.rm = ManagedSparseRewardMachine(rm_file)
        self.aux_node = self.rm.u0
        self.start_nodes = []
        for event in self.rm.delta_u[self.aux_node]:
            self.start_nodes.append(self.rm.delta_u[self.aux_node][event])
        # print(self.start_nodes)
        self.event_list = event_list
        self.set_list = set_list
        # print(self.set_list)
        self.curr_assignment = list(np.random.permutation([i for i in range(num_agents)]))
        self.assignment_method = assignment_method
        self.num_agents = num_agents

        


    def assign(self, agent_list):
        permutation_qs = self.calculate_permutation_qs(agent_list)
        if self.assignment_method == "ground_truth":
            self.curr_assignment = [0,1,2]
        elif self.assignment_method == "random": 
            self.curr_assignment = list(random.choice(list(permutation_qs.keys())))
        elif self.assignment_method == "greedy":
            self.curr_assignment = list(max(permutation_qs, key=permutation_qs.get))
        else:
            raise Exception("STUPID ASS MF")
        
        for i in range(len(agent_list)):
            i_assigned = self.curr_assignment[i]
            agent_list[i].u = self.start_nodes[i_assigned]
            agent_list[i].is_task_complete = 0

        # 1) Take random batch of exp, evaluate each agent's policy across all assignments, see which assignment gets highest reward
        # 


        # # Random assigning
        # for i, new_id in enumerate(np.random.permutation(len(agent_list))):
        #     agent_list[i].agent_id = new_id # to fix the hard coded stuff
        #     agent_list[i].u = self.start_nodes[new_id]  # change start node to new_id
        #     agent_list[i].is_task_complete = 0

    def load_assignment(self, agent_list):
        for i in range(len(agent_list)):
            i_assigned = self.curr_assignment[i]
            agent_list[i].u = self.start_nodes[i_assigned]
            agent_list[i].is_task_complete = 0


    # def get_subtask_states(self, i):
    #     return self.set_list[i]
    
    def initial_state(self, i):
        return self.start_nodes[self.curr_assignment[i]]

    def get_events(self, agent_id):
        return self.event_list[self.curr_assignment[agent_id]]
        # return self.event_list[agent_id]


    def calculate_permutation_qs(self, agent_list):
        res = {}
        for permutation in itertools.permutations(list(range(self.num_agents))):
            q_sum = 0
            for i in range(len(permutation)):
                starting_rm_state = self.start_nodes[permutation[i]]
                curr_state = np.row_stack(([agent_list[i].s_i], [starting_rm_state])).T
                qa = agent_list[i].Q(ptu.from_numpy(curr_state).float())
                q = torch.max(qa).item()
                q_sum += q
            
            res[tuple(permutation)] = q_sum
        return res
                
                

    