from reward_machines.managed_sparse_reward_machine import ManagedSparseRewardMachine
class Manager:
    def __init__(self, rm_file, set_list, event_list):
        self.rm = ManagedSparseRewardMachine(rm_file)
        self.aux_node = self.rm.u0
        self.start_nodes = []
        for event in self.rm.delta_u[self.aux_node]:
            self.start_nodes.append(self.rm.delta_u[self.aux_node][event])
        # print(self.start_nodes)
        self.event_list = event_list
        self.set_list = set_list
        # print(self.set_list)


    def assign(self, agent_list):
        for i in range(len(agent_list)):
            agent_list[i].u = self.start_nodes[i]  # random assignment for now
            agent_list[i].is_task_complete = 0

    def get_subtask_states(self, i):
        return self.set_list[i]
    
    def initial_state(self, i):
        return self.start_nodes[i]

    def get_events(self, agent_id):
        return self.event_list[agent_id]

    