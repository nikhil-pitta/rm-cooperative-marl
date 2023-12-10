
import wandb
from datetime import datetime
import argparse



def lpdbuttons_sweep():

    from buttons_config import buttons_config
    from experiments.ddqprm_lp import run_multi_agent_experiment

    wandb.init(project="lpdbuttons")
    num_times = 1 # Number of separate trials to run the algorithm for

    num_agents = 3 # This will be automatically set to 3 for buttons experiment (max 10)

    config = dict(wandb.config)
    tester = buttons_config(num_times, num_agents)

    tester.exploration_fraction = config['eps_decay']
    tester.config = config

    run_multi_agent_experiment(tester, num_agents, num_times, config['batch_size'], config['buffer_size'], "multiply", show_print_1=False)
    # wandb.log({"total_reward": total_reward})

def lpdbuttons_hard_sweep():

    from hard_buttons_config import hard_buttons_config
    from experiments.ddqprm_lp import run_multi_agent_experiment

    wandb.init(project="lpdbuttons_hard")
    num_times = 1 # Number of separate trials to run the algorithm for

    num_agents = 3 # This will be automatically set to 3 for buttons experiment (max 10)

    config = dict(wandb.config)
    tester = hard_buttons_config(num_times, num_agents)

    tester.exploration_fraction = config['eps_decay']
    tester.config = config

    run_multi_agent_experiment(tester, num_agents, num_times, config['batch_size'], config['buffer_size'], "multiply", show_print_1=False)
    # wandb.log({"total_reward": total_reward})


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Multi-agent experiment runner.")
    parser.add_argument("--experiment_name", type=str, default="lpdbuttons", 
                        help="Name of the experiment")
    
    args = parser.parse_args()
    experiment = args.experiment_name

    if experiment == "lpdbuttons":
        train = lpdbuttons_sweep
    elif experiment == "lpdbuttons_hard":
        train = lpdbuttons_hard_sweep
    else:
        raise Exception("INVALID")

    # using wandb sweep
    sweep_configuration = {
        "method": "random",
        "metric": {"goal": "maximize", "name": "Episode Reward"},
        "parameters": {
            "batch_size": {"values": [512]},
            "buffer_size": {"values": [5000]},
            "eps_decay": {"values": [0.98]},
            "learning_rate": {"values": [0.001]},
            "adam_betas": {"values": [0.9]},
            "num_layers": {"values": [2, 3]},
            "layer_size": {"values": [64]},
            "cheat_thresh": {"values": [0.3]}
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project=experiment)
    wandb.agent(sweep_id, function=train)