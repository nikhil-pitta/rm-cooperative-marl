from tester.tester import Tester
from tester.tester_params import TestingParameters
from tester.learning_params import LearningParameters
import os

def fair_buttons_config(num_times, num_agents):
    """
    Function setting the experiment parameters and environment.

    Output
    ------
    Tester : tester object
        Object containing the information necessary to run this experiment.
    """
    base_file_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    joint_rm_file = os.path.join(base_file_path, 'experiments', 'buttons_fair', 'team_buttons_fair_rm.txt')

    local_rm_files = []
    for i in range(num_agents):
        local_rm_string = os.path.join(base_file_path, 'experiments', 'buttons_fair', 'buttons_rm_agent_{}.txt'.format(i+1))
        local_rm_files.append(local_rm_string)

    step_unit = 1000

    # configuration of testing params
    testing_params = TestingParameters()
    testing_params.test = True
    testing_params.test_freq = 1*step_unit 
    testing_params.num_steps = step_unit

    # configuration of learning params
    learning_params = LearningParameters()
    learning_params.gamma = 0.8 # 0.9
    learning_params.alpha = 0.8
    learning_params.T = 50
    # learning_params.initial_epsilon = 0.0 # Set epsilon to zero to turn off epsilon-greedy exploration (only using boltzmann)

    ######### for deepqbuttons #############
    learning_params.initial_epsilon = 1
    learning_params.exploration_fraction = 0.98
    ######### for deepqbuttons #############


    learning_params.max_timesteps_per_task = testing_params.num_steps

    tester = Tester(learning_params, testing_params)
    tester.step_unit = step_unit
    tester.total_steps = 250 * step_unit # 100 * step_unit
    tester.min_steps = 1
    tester.early_stopping_point = 250 * step_unit
    # tester.total_trajs = 250 * 

    tester.num_times = num_times
    tester.num_agents = num_agents

    tester.rm_test_file = joint_rm_file
    tester.rm_learning_file_list = local_rm_files

    # Set the environment settings for the experiment
    env_settings = dict()
    env_settings['Nr'] = 10
    env_settings['Nc'] = 10
    env_settings['initial_states'] = [5, 5, 5]
    # env_settings['walls'] = [(0, 3), (1, 3), (2, 3), (3,3), (4,3),
    #                             (0,7), (1,7), (2,7), (3,7), (4,7)]
    # env_settings['goal_location'] = (6,5)
    env_settings['yellow_button'] = (6,2)
    env_settings['green_button'] = (6,5)
    env_settings['red_button'] = (6,8)

    env_settings['p'] = 0.98

    tester.env_settings = env_settings

    tester.experiment = 'buttons_fair'

    return tester