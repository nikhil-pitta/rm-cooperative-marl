
import wandb
from datetime import datetime


from buttons_config import buttons_config
from experiments.ddqprm_lp import run_multi_agent_experiment

def train():
    wandb.init(project="lpdbuttons_sweep")
    num_times = 1 # Number of separate trials to run the algorithm for

    num_agents = 3 # This will be automatically set to 3 for buttons experiment (max 10)

    config = dict(wandb.config)
    tester = buttons_config(num_times, num_agents)

    tester.exploration_fraction = config['eps_decay']
    tester.config = config

    run_multi_agent_experiment(tester, num_agents, num_times, config['batch_size'], config['buffer_size'], "multiply", show_print_1=False)
    # wandb.log({"total_reward": total_reward})

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

sweep_id = wandb.sweep(sweep=sweep_configuration, project="lpdbuttons_sweep")
wandb.agent(sweep_id, function=train)