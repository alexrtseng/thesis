import wandb


def start_price_node_sweep(
    sweep_config,
):
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="Thesis",
    )

    sweep_id
