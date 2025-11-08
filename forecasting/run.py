from forecasting.sweep_runner import WANDB_PROJECT_NAME, run_all_models_for_node


def run_all_basic_sweeps(pnode_id: int):
    run_all_models_for_node(
        pnode_id=pnode_id,
        project=WANDB_PROJECT_NAME,
        count=1,
        subset_data_size=0.01,
    )


if __name__ == "__main__":
    run_all_basic_sweeps(2156113094)
