import wandb


def start_price_node_sweep(sweep_config):
    """Start a W&B sweep and return its ID."""
    sweep_id = wandb.sweep(sweep=sweep_config, project="Thesis")
    return sweep_id
