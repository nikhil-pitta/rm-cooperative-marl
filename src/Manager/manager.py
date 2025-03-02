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
        self.curr_permutation_qs = {}
        self.epsilon = 1
        self.epsilon_decay = 0.9

        ### UCB Specific ####

        self.permutation_counts = {perm: 0 for perm in itertools.permutations(range(num_agents))}
        self.permutation_total_rewards = {perm: 0.0 for perm in itertools.permutations(range(num_agents))}
        # print("HELLO", self.permutation_counts)
        self.total_selections = 0

        # UCB exploration parameter
        self.ucb_c = 1.5



    def assign(self, agent_list):
        self.curr_permutation_qs = self.calculate_permutation_qs(agent_list, True)

        if self.assignment_method == "ground_truth":
            self.curr_assignment = [0,1,2]
        elif self.assignment_method == "random": 
            self.curr_assignment = list(random.choice(list(self.curr_permutation_qs.keys())))
        elif self.assignment_method == "epsilon_greedy":
            self.curr_permutation_qs = self.calculate_permutation_qs(agent_list)

            if random.random() < self.epsilon:
                self.curr_assignment = list(random.choice(list(self.curr_permutation_qs.keys())))
            else:
                self.curr_assignment = list(max(self.curr_permutation_qs, key=self.curr_permutation_qs.get))
            self.epsilon *= self.epsilon_decay
        elif self.assignment_method == "multiply":
            self.curr_permutation_qs = self.calculate_permutation_qs(agent_list, True)

            if random.random() < self.epsilon:
                self.curr_assignment = list(random.choice(list(self.curr_permutation_qs.keys())))
            else:
                self.curr_assignment = list(max(self.curr_permutation_qs, key=self.curr_permutation_qs.get))
            self.epsilon *= self.epsilon_decay

        elif self.assignment_method == "UCB":

            # OVERRIDE WITH UCB SCORES
            ucb_values = {perm: self.calculate_ucb_value(perm) for perm in self.permutation_counts.keys()}
            self.curr_permutation_qs = ucb_values


            if self.total_selections < len(self.permutation_counts):
                # Ensure each permutation is selected at least once in the beginning
                self.curr_assignment = list(self.permutation_counts.keys())[self.total_selections]
            else:
                # Calculate UCB value for each permutation and select the one with the highest UCB value
                # ucb_values = {perm: self.calculate_ucb_value(perm) for perm in self.permutation_counts.keys()}
                self.curr_assignment = list(max(ucb_values, key=ucb_values.get))

            # Update counts and total selections
            perm_tuple = tuple(self.curr_assignment)
            self.permutation_counts[perm_tuple] += 1
            self.total_selections += 1
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

    def calculate_permutation_qs(self, agent_list, multiply=False):
        res = {}
        for permutation in itertools.permutations(list(range(self.num_agents))):
            accumulator = 1 if multiply else 0

            for i in range(len(permutation)):
                starting_rm_state = self.start_nodes[permutation[i]]
                curr_state = np.row_stack(([agent_list[i].s_i], [starting_rm_state])).T
                qa = agent_list[i].Q(ptu.from_numpy(curr_state).float())
                q = torch.max(qa).item()

                if multiply:
                    accumulator *= q
                else:
                    accumulator += q
            
            res[tuple(permutation)] = accumulator
        return res
    
    
    ### FOR UCB ###
    def update_rewards(self, permutation, reward):
        # Update the total reward for a permutation after an assignment is completed
        self.permutation_total_rewards[tuple(permutation)] += reward
    
    def calculate_ucb_value(self, permutation):
        # Calculate the UCB value for a given permutation
        if self.permutation_counts[permutation] == 0:
            return float('inf')  # Represents a strong incentive to select this permutation
        
        average_reward = self.permutation_total_rewards[permutation] / self.permutation_counts[permutation]
        confidence = np.sqrt((2 * np.log(self.total_selections)) / self.permutation_counts[permutation])
        return average_reward + self.ucb_c * confidence
                    
                

    