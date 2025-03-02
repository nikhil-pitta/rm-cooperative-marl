import wandb
import numpy as np
import random, time
import copy

from tester.tester import Tester
from Agent.teamwork_agent import Agent
from Environments.coop_buttons.buttons_env_teamwork import ButtonsEnv
from Environments.coop_buttons.multi_agent_buttons_env import MultiAgentButtonsEnv
from Environments.rendezvous.gridworld_env import GridWorldEnv
from Environments.rendezvous.multi_agent_gridworld_env import MultiAgentGridWorldEnv
from Environments.buttons_hard.buttons_hard import HardButtonsEnv
from Environments.buttons_hard.multi_agent_buttons_hard import HardMultiAgentButtonsEnv
from Environments.buttons_fair.buttons_fair import FairButtonsEnv
from Environments.buttons_fair.multi_agent_buttons_fair import FairMultiAgentButtonsEnv
from Environments.buttons_all.buttons_all import AllButtonsEnv
from Environments.buttons_all.multi_agent_buttons_all import AllMultiAgentButtonsEnv
import matplotlib.pyplot as plt
import infrastructure.rm_utils as rm_builder
from Manager.manager import Manager
import itertools

from tqdm import tqdm

def run_qlearning_task(epsilon,
                        tester,
                        agent_list,
                        batch_size,
                        buffer_size,
                        manager,
                        train_reward_buildup,
                        train_discount_reward_buildup,
                        show_print=True, num_iters=5):
    """
    This code runs one q-learning episode. q-functions, and accumulated reward values of agents
    are updated accordingly. If the appropriate number of steps have elapsed, this function will
    additionally run a test episode.

    Parameters
    ----------
    epsilon : float
        Numerical value in (0,1) representing likelihood of choosing a random action.
    tester : Tester object
        Object containing necessary information for current experiment.
    agent_list : list of Agent objects
        Agent objects to be trained and tested.
    show_print : bool
        Optional flag indicating whether or not to print output statements to terminal.
    """
    # Initializing parameters and the game
    learning_params = tester.learning_params
    testing_params = tester.testing_params

    num_agents = len(agent_list)
    manager.assign(agent_list)
    for i in range(num_agents):
        agent_list[i].reset_state()
        # print(f"agent start {i}", agent_list[i].u)
        # agent_list[i].initialize_reward_machine()

    num_steps = learning_params.max_timesteps_per_task

    # Load the appropriate environments for training
    if tester.experiment == 'rendezvous':
        training_environments = []
        for i in range(num_agents):
            training_environments.append(GridWorldEnv(agent_list[i].rm_file, i+1, tester.env_settings))
    if tester.experiment == 'buttons':
        training_environments = []
        for i in range(num_agents):
            training_environments.append(ButtonsEnv(agent_list[i].rm_file, i+1, tester.env_settings, manager, tester.config))
    if tester.experiment == "buttons_hard":
        training_environments = []
        for i in range(num_agents):
            training_environments.append(HardButtonsEnv(agent_list[i].rm_file, i+1, tester.env_settings, manager))
    if tester.experiment == "buttons_fair":
        training_environments = []
        for i in range(num_agents):
            training_environments.append(FairButtonsEnv(agent_list[i].rm_file, i+1, tester.env_settings, manager))
    if tester.experiment == "buttons_all":
        training_environments = []
        for i in range(num_agents):
            training_environments.append(AllButtonsEnv(agent_list[i].rm_file, i+1, tester.env_settings, manager))
    broke_loop = False

    for t in range(num_steps):
        # Update step count
        tester.add_step()

        for i in range(num_agents):
            # Perform a q-learning step.

            if not(agent_list[i].is_task_complete):
                current_u = agent_list[i].u
                # print(f"agent {i} at state {current_u}")
                s, a = agent_list[i].get_next_action(epsilon, learning_params)
                r, l, s_new = training_environments[i].environment_step(s,a)

                u2 = training_environments[i].u
                # a = training_environments[i].get_last_action() # due to MDP slip
                # # agent_list[i].update_agent(s_new, a, r, l, learning_params)
                # if tester.get_current_step() > agent_list[i].buffer.max_: 
                #     agent_list[i].update_agent(s_new, a, r, l, learning_params, tester.get_current_step())
                agent_list[i].buffer.add(s, current_u, a, r, s_new, u2)

                for u in agent_list[i].rm.U:
                    if not (u == current_u) and not (u in agent_list[i].rm.T) and not (u == agent_list[i].rm.u0):
                    # if not (u == current_u) and not (u in agent_list[i].rm.T):
                        new_l = training_environments[i].get_mdp_label(s, s_new, u)
                        new_r = 0
                        u_temp = u
                        u2 = u
                        for e in new_l:
                            # Get the new reward machine state and the reward of this step
                            u2 = agent_list[i].rm.get_next_state(u_temp, e)
                            new_r = new_r + agent_list[i].rm.get_reward(u_temp, u2)
                            # Update the reward machine state
                            u_temp = u2
                        # agent_list[i].update_q_function(s, s_new, u, u2, a, r, learning_params)
                        # if tester.get_current_step() > agent_list[i].buffer.max_: 
                        #     agent_list[i].update_q_function(s, s_new, u, u2, a, r, learning_params, tester.get_current_step())
                        agent_list[i].buffer.add(s, u, a, new_r, s_new, u2)

            if tester.get_current_step() > agent_list[i].buffer.max_: 
                agent_list[i].update_agent(s_new, r, l, learning_params, tester.get_current_step())
            else:
                agent_list[i].update_agent_rm(l)

        # If enough steps have elapsed, test and save the performance of the agents.
        if testing_params.test and tester.get_current_step() % testing_params.test_freq == 0:
            # print("TEST")

            t_init = time.time()
            step = tester.get_current_step()

            agent_list_copy = []

            # Need to create a copy of the agent for testing. If we pass the agent directly
            # mid-episode to the test function, the test will reset the world-state and reward machine 
            # state before the training episode has been completed.
            for i in range(num_agents):
                rm_file = agent_list[i].rm_file
                s_i = agent_list[i].s_i
                actions = agent_list[i].actions
                agent_id = agent_list[i].agent_id
                num_states = agent_list[i].num_states
                agent_copy = Agent(rm_file, s_i, num_states, actions, agent_id, batch_size, buffer_size, tester, manager)
                # Pass only the q-function by reference so that the testing updates the original agent's q-function.
                # agent_copy.q = agent_list[i].q
                agent_copy.Q = copy.deepcopy(agent_list[i].Q)
                agent_copy.Q_target = copy.deepcopy(agent_list[i].Q_target)

                agent_list_copy.append(agent_copy)

            # Run a test of the performance of the agents
            total_testing_reward = 0 
            total_testing_steps = 0
            total_agent_reward = [0]*num_agents
            total_testing_discounted_reward = 0

            for n in range(num_iters):
                testing_reward, testing_discounted_reward, trajectory, testing_steps = run_multi_agent_qlearning_test(agent_list_copy,
                                                                                            tester,
                                                                                            learning_params,
                                                                                            testing_params,
                                                                                            manager,
                                                                                            show_print = show_print and (n == num_iters - 1))
                total_testing_reward += testing_reward
                total_testing_steps += testing_steps
                total_testing_discounted_reward += testing_discounted_reward

                for k in range(num_agents):
                    total_agent_reward[k] += int(agent_list_copy[k].is_task_complete)

            wandb.log({'Episode Reward': (total_testing_reward + sum(train_reward_buildup))/ (num_iters + len(train_reward_buildup)), 
                       'Discounted Episode Reward': (total_testing_discounted_reward + sum(train_discount_reward_buildup)) / (num_iters+len(train_discount_reward_buildup)),
                       "Episode Epsilon": epsilon, 
                       'Number of Steps Reward Achieved': total_testing_steps/num_iters, 
                       "Test Trajectory": tester.get_global_step()})
            # if (total_testing_discounted_reward + sum(train_discount_reward_buildup)) / (num_iters+len(train_discount_reward_buildup)) >= 0.24:
            #     tester.early_stopping_point = -1
            train_reward_buildup.clear()
            train_discount_reward_buildup.clear()
            
            for a_n in range(num_agents):
                # if total_agent_reward[a_n] > 0:
                #     print(a_n, manager.curr_assignment, agent_list_copy[a_n].local_event_set)
                wandb.log({f"Reward Achieved for Agent {a_n}": total_agent_reward[a_n]/num_iters, "Test Trajectory": tester.get_global_step()})
                wandb.log({f"Critic Loss for Agent {a_n}": agent_list[a_n].curr_loss, "Test Trajectory": tester.get_global_step()})

            permutation_q_log = {f"{permutation}": manager.curr_permutation_qs[permutation] for permutation in itertools.permutations(list(range(num_agents)))}
            permutation_q_log["Test Trajectory"] = tester.get_global_step()
            wandb.log(permutation_q_log)

            tester.add_global()

            epsilon *= learning_params.exploration_fraction
            
            # Save the testing reward
            if 0 not in tester.results.keys():
                tester.results[0] = {}
            if step not in tester.results[0]:
                tester.results[0][step] = []
            tester.results[0][step].append(testing_reward)

            # Save the testing trace
            if 'trajectories' not in tester.results.keys():
                tester.results['trajectories'] = {}
            if step not in tester.results['trajectories']:
                tester.results['trajectories'][step] = []
            tester.results['trajectories'][step].append(trajectory)

            # Save how many steps it took to complete the task
            if 'testing_steps' not in tester.results.keys():
                tester.results['testing_steps'] = {}
            if step not in tester.results['testing_steps']:
                tester.results['testing_steps'][step] = []
            tester.results['testing_steps'][step].append(testing_steps)

            # Keep track of the steps taken
            if len(tester.steps) == 0 or tester.steps[-1] < step:
                tester.steps.append(step)
        
        # If the agents has completed its task, reset it to its initial state.
        if all(agent.is_task_complete for agent in agent_list):
            train_discount_reward_buildup.append(learning_params.gamma ** (t-1))
            train_reward_buildup.append(1)
            # FOR UCB, updating trajectory reward
            manager.update_rewards(manager.curr_assignment, 1)

            manager.assign(agent_list)
            for i in range(num_agents):
                agent_list[i].reset_state()
                # print(f"agent start {i}", agent_list[i].u)
                # agent_list[i].initialize_reward_machine()
            
            # Make sure we've run at least the minimum number of training steps before breaking the loop
            if tester.stop_task(t):
                broke_loop = True
                break

        # checking the steps time-out
        if tester.stop_learning():
            train_reward_buildup.append(0)
            train_discount_reward_buildup.append(0)
            broke_loop = True
            break
    
    if not broke_loop:
        train_reward_buildup.append(0)
        train_discount_reward_buildup.append(0)
    return epsilon

def run_multi_agent_qlearning_test(agent_list,
                                    tester,
                                    learning_params,
                                    testing_params,
                                    manager,
                                    show_print=True):
    """
    Run a test of the q-learning with reward machine method with the current q-function. 

    Parameters
    ----------
    agent_list : list of Agent objects
        Agent objects to be trained and tested.
    learning_params : LearningParameters object
        Object storing parameters to be used in learning.
    Testing_params : TestingParameters object
        Object storing parameters to be used in testing.

    Ouputs
    ------
    testing_reard : float
        Reward achieved by agent during this test episode.
    trajectory : list
        List of dictionaries containing information on current step of test.
    step : int
        Number of testing steps required to complete the task.
    """
    num_agents = len(agent_list)

    if tester.experiment == 'rendezvous':
        testing_env = MultiAgentGridWorldEnv(tester.rm_test_file, num_agents, tester.env_settings)
    if tester.experiment == 'search_and_rescue':
        target_region = np.random.choice([0,1,2,3,4])
        testing_env = SearchAndRescueEnv(tester.rm_test_file, target_region, tester.env_settings)
    if tester.experiment == 'buttons':
        testing_env = MultiAgentButtonsEnv(tester.rm_test_file, num_agents, tester.env_settings)
    if tester.experiment == 'buttons_hard':
        testing_env = HardMultiAgentButtonsEnv(tester.rm_test_file, num_agents, tester.env_settings)
    if tester.experiment == "buttons_fair":
        testing_env = FairMultiAgentButtonsEnv(tester.rm_test_file, num_agents, tester.env_settings)
    if tester.experiment == "buttons_all":
        testing_env = AllMultiAgentButtonsEnv(tester.rm_test_file, num_agents, tester.env_settings)
    manager.load_assignment(agent_list)
    for i in range(num_agents):
        agent_list[i].reset_state()
        # print(f"agent start {i}", agent_list[i].u)
        # agent_list[i].initialize_reward_machine()

    s_team = np.full(num_agents, -1, dtype=int)
    for i in range(num_agents):
        s_team[i] = agent_list[i].s
    a_team = np.full(num_agents, -1, dtype=int)
    # u_team = np.full(num_agents, -1, dtype=int)
    # for i in range(num_agents):
    #     u_team[i] = agent_list[i].u
    testing_reward = 0
    testing_discounted_reward = 0

    trajectory = []

    step = 0

    # Starting interaction with the environment
    for t in range(testing_params.num_steps):
        step = step + 1

        # Perform a team step
        for i in range(num_agents):
            s, a = agent_list[i].get_next_action(-1.0, learning_params)
            s_team[i] = s
            a_team[i] = a
            # u_team[i] = agent_list[i].u

        # trajectory.append({'s' : np.array(s_team, dtype=int), 'a' : np.array(a_team, dtype=int), 'u_team': np.array(u_team, dtype=int), 'u': int(testing_env.u)})

        r, l, s_team_next = testing_env.environment_step(s_team, a_team)
        trajectory.append(s_team_next)

        testing_reward = testing_reward + r
        testing_discounted_reward += (learning_params.gamma ** (t-1)) * r

        projected_l_dict = {}
        for i in range(num_agents):
            # print(agent_list[i].local_event_set, set(l))
            # Agent i's projected label is the intersection of l with its local event set
            projected_l_dict[i] = list(set(agent_list[i].local_event_set) & set(l))
            # print(f"aggent {i}", set(agent_list[i].local_event_set))
            # Check if the event causes a transition from the agent's current RM state
            if not(agent_list[i].is_local_event_available(projected_l_dict[i])):
                projected_l_dict[i] = []

        for i in range(num_agents):
            # Enforce synchronization requirement on shared events
            if projected_l_dict[i]:
                for event in projected_l_dict[i]:
                    for j in range(num_agents):
                        if (event in set(agent_list[j].local_event_set)) and (not (projected_l_dict[j] == projected_l_dict[i])):
                            projected_l_dict[i] = []

            # update the agent's internal representation
            # a = testing_env.get_last_action(i)
            # print("projection dict", projected_l_dict[i])
            agent_list[i].update_agent(s_team_next[i], r, projected_l_dict[i], learning_params, tester.get_current_step(), update_q_function=False, evaluate_critic_loss=True)

        # for i in range(num_agents):
        #     wandb.log({f"Reward Achieved for Agent {i}": int(agent_list[i].is_task_complete), "Step": tester.get_global_step()})

        if all(agent.is_task_complete for agent in agent_list):
            break

    # for i in range(num_agents):
    #     wandb.log({f"Reward Achieved for Agent {i}": int(agent_list[i].is_task_complete), "Test Trajectory": tester.get_global_step()})

    if show_print:
        testing_env.log_traj(trajectory, tester.get_global_step())
    
    print('Reward of {} achieved in {} steps. Current Trajectory: {} of {}'.format(testing_reward, step, tester.current_step, tester.total_steps))



    return testing_reward, testing_discounted_reward, trajectory, step

def run_multi_agent_experiment(tester,num_agents,num_times,batch_size, buffer_size, assignment_method, show_print_1=True):
    """
    Run the entire q-learning with reward machines experiment a number of times specified by num_times.

    Inputs
    ------
    tester : Tester object
        Test object holding true reward machine and all information relating
        to the particular tasks, world, learning parameters, and experimental results.
    num_agents : int
        Number of agents in this experiment.
    num_times : int
        Number of times to run the entire experiment (restarting training from scratch).
    show_print : bool
        Flag indicating whether or not to output text to the terminal.
    """

    learning_params = tester.learning_params

    for t in range(num_times):
        # Reseting default step values
        tester.restart()

        rm_test_file = tester.rm_test_file
        rm_learning_file_list = tester.rm_learning_file_list
        joined_rm_file, set_list, event_list = rm_builder.build_rm(rm_learning_file_list)

        # Verify that the number of local reward machines matches the number of agents in the experiment.
        assertion_string = "Number of specified local reward machines must match specified number of agents."
        assert (len(tester.rm_learning_file_list) == num_agents), assertion_string

        if tester.experiment == 'rendezvous':
            testing_env = MultiAgentGridWorldEnv(tester.rm_test_file, num_agents, tester.env_settings)
            num_states = testing_env.num_states
        if tester.experiment == 'buttons':
            testing_env = MultiAgentButtonsEnv(tester.rm_test_file, num_agents, tester.env_settings)
            num_states = testing_env.num_states
        if tester.experiment == 'buttons_hard':
            testing_env = HardMultiAgentButtonsEnv(tester.rm_test_file, num_agents, tester.env_settings)
            num_states = testing_env.num_states
        if tester.experiment == "buttons_fair":
            testing_env = FairMultiAgentButtonsEnv(tester.rm_test_file, num_agents, tester.env_settings)
            num_states = testing_env.num_states
        if tester.experiment == "buttons_all":
            testing_env = AllMultiAgentButtonsEnv(tester.rm_test_file, num_agents, tester.env_settings)
            num_states = testing_env.num_states


        manager = Manager(joined_rm_file, set_list, event_list, 3, assignment_method)

        # Create the a list of agents for this experiment
        agent_list = [] 
        for i in range(num_agents):
            actions = testing_env.get_actions(i)
            s_i = testing_env.get_initial_state(i)
            agent_list.append(Agent(joined_rm_file, s_i, num_states, actions, i, batch_size, buffer_size, tester, manager))

        num_episodes = 0
        step = 0

        # Task loop
        epsilon = learning_params.initial_epsilon
        train_reward_buildup = []
        train_discount_reward_buildup = []
        while not tester.stop_learning():
            # num_episodes += 1
            # epsilon = epsilon*learning_params.exploration_fraction
            # print("TRAIN")
            epsilon = run_qlearning_task(epsilon,
                                tester,
                                agent_list,
                                batch_size, 
                                buffer_size,
                                manager,
                                train_reward_buildup,
                                train_discount_reward_buildup,
                                show_print=show_print_1)

        # for _ in tqdm(range(tester.total_steps)):
        #     epsilon = run_qlearning_task(epsilon,
        #                         tester,
        #                         agent_list,
        #                         batch_size, 
        #                         buffer_size,
        #                         manager,
        #                         show_print=show_print_1)

            

        # Backing up the results
        print('Finished iteration ',t)

    tester.agent_list = agent_list

    # plot_multi_agent_results(tester, num_agents)

def plot_multi_agent_results(tester, num_agents):
    """
    Plot the results stored in tester.results for each of the agents.
    """

    prc_25 = list()
    prc_50 = list()
    prc_75 = list()

    # Buffers for plots
    current_step = list()
    current_25 = list()
    current_50 = list()
    current_75 = list()
    steps = list()

    plot_dict = tester.results['testing_steps']

    for step in plot_dict.keys():
        if len(current_step) < 10:
            current_25.append(np.percentile(np.array(plot_dict[step]),25))
            current_50.append(np.percentile(np.array(plot_dict[step]),50))
            current_75.append(np.percentile(np.array(plot_dict[step]),75))
            current_step.append(sum(plot_dict[step])/len(plot_dict[step]))
        else:
            current_step.pop(0)
            current_25.pop(0)
            current_50.pop(0)
            current_75.pop(0)
            current_25.append(np.percentile(np.array(plot_dict[step]),25))
            current_50.append(np.percentile(np.array(plot_dict[step]),50))
            current_75.append(np.percentile(np.array(plot_dict[step]),75))
            current_step.append(sum(plot_dict[step])/len(plot_dict[step]))

        prc_25.append(sum(current_25)/len(current_25))
        prc_50.append(sum(current_50)/len(current_50))
        prc_75.append(sum(current_75)/len(current_75))
        steps.append(step)

    plt.plot(steps, prc_25, alpha=0)
    plt.plot(steps, prc_50, color='red')
    plt.plot(steps, prc_75, alpha=0)
    plt.grid()
    plt.fill_between(steps, prc_50, prc_25, color='red', alpha=0.25)
    plt.fill_between(steps, prc_50, prc_75, color='red', alpha=0.25)
    plt.ylabel('Testing Steps to Task Completion', fontsize=15)
    plt.xlabel('Training Steps', fontsize=15)
    plt.locator_params(axis='x', nbins=5)

    plt.show()

