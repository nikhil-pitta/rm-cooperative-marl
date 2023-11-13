
import wandb
import pickle
from datetime import datetime
import os
import argparse

if __name__ == "__main__":

    num_times = 10 # Number of separate trials to run the algorithm for

    num_agents = 3 # This will be automatically set to 3 for buttons experiment (max 10)

    parser = argparse.ArgumentParser(description="Multi-agent experiment runner.")
    parser.add_argument("--experiment_name", type=str, default="dbuttons", 
                        help="Name of the experiment")
    args = parser.parse_args()
    
    experiment = args.experiment_name

    wandb.init(project = experiment)
    wandb.define_metric("Step")
    wandb.define_metric("Episode Reward", step_metric="Step")
    wandb.define_metric("Episode Epsilon", step_metric="Step")
    wandb.define_metric("Number of Steps Reward Achieved", step_metric="Step")
    for i in range(num_agents):
        wandb.define_metric(f"Reward Achieved for Agent {i}", step_metric="Step")
        wandb.define_metric(f"Critic Loss for Agent {i}", step_metric="Step")


    if experiment == 'rendezvous':
        from rendezvous_config import rendezvous_config
        from experiments.dqprm import run_multi_agent_experiment
        tester = rendezvous_config(num_times, num_agents) # Get test object from config script
        run_multi_agent_experiment(tester, num_agents, num_times)

    if experiment == 'drendezvous':
        from rendezvous_config import rendezvous_config
        from experiments.ddqprm import run_multi_agent_experiment
        tester = rendezvous_config(num_times, num_agents)
        run_multi_agent_experiment(tester, num_agents, num_times, 128, 200)

    if experiment == 'centralized_rendezvous':
        from rendezvous_config import rendezvous_config
        from experiments.run_centralized_coordination_experiment import run_centralized_experiment
        tester = rendezvous_config(num_times, num_agents)
        run_centralized_experiment(tester, num_agents, num_times, show_print=True)

    if experiment == 'ihrl_rendezvous':
        from rendezvous_config import rendezvous_config
        from experiments.run_ihrl_experiment import run_ihrl_experiment
        tester = rendezvous_config(num_times, num_agents)
        run_ihrl_experiment(tester, num_agents, num_times, show_print=True)

    if experiment == 'iql_rendezvous':
        from rendezvous_config import rendezvous_config
        from experiments.iql import run_iql_experiment
        tester = rendezvous_config(num_times, num_agents)
        run_iql_experiment(tester, num_agents, num_times, show_print=True)

    if experiment == 'buttons':
        from buttons_config import buttons_config
        from experiments.dqprm import run_multi_agent_experiment
        num_agents = 3 # Num agents must be 3 for this example
        tester = buttons_config(num_times, num_agents) # Get test object from config script
        run_multi_agent_experiment(tester, num_agents, num_times, show_print=True)
    
    if experiment == "dbuttons":
        from buttons_config import buttons_config
        from experiments.ddqprm import run_multi_agent_experiment
        num_agents = 3 # Num agents must be 3 for this example
        tester = buttons_config(num_times, num_agents) # Get test object from config script
        run_multi_agent_experiment(tester, num_agents, num_times, 512, 5000)

    if experiment == 'ihrl_buttons':
        from buttons_config import buttons_config
        from experiments.run_ihrl_experiment import run_ihrl_experiment
        num_agents = 3 # Num agents must be 3 for this example
        tester = buttons_config(num_times, num_agents)
        run_ihrl_experiment(tester, num_agents, num_times, show_print=True)
    
    if experiment == 'iql_buttons':
        from buttons_config import buttons_config
        from experiments.iql import run_iql_experiment
        num_agents = 3 # Num agents must be 3 for this example
        tester = buttons_config(num_times, num_agents)
        run_iql_experiment(tester, num_agents, num_times, show_print=True)


    # Save the results
    parentDir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    data_path = os.path.join(parentDir, 'data')
    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    experiment_data_path = os.path.join(data_path, experiment)

    if not os.path.isdir(experiment_data_path):
        os.mkdir(experiment_data_path)

    now = datetime.now()
    save_file_str = r'\{}_'.format(now.strftime("%Y-%m-%d_%H-%M-%S"))
    save_file_str = save_file_str + experiment + '.p'
    save_file = open(experiment_data_path + save_file_str, "wb")
    pickle.dump(tester, save_file)


