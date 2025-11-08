import wandb
from pytorch_lightning.callbacks import Callback


def wandb_logger():
    """
    A callback to log training and validation loss to Weights & Biases. Also
    keeps track of the model's gradients and parameters, and artifcacts the
    best runs.
    Args:
        None
    Returns:
        Instance of WandbLogger
    """

    class WandbLogger(Callback):
        def __init__(self, metrics_to_log=None):
            self.metrics_to_log = metrics_to_log or ["train_loss", "val_loss"]

        def on_train_start(self, trainer, pl_module):
            print(f"Watching {pl_module}")
            wandb.watch(pl_module, log="all", log_freq=1)

        def on_train_epoch_end(self, trainer, pl_module):
            if self.metrics_to_log is None:
                metrics = {
                    key: float(value) for key, value in trainer.callback_metrics.items()
                }
            else:
                metrics = {
                    key: float(value)
                    for key, value in trainer.callback_metrics.items()
                    if key in self.metrics_to_log
                }
            wandb.log(metrics)

        def on_validation_epoch_end(self, trainer, pl_module):
            if self.metrics_to_log is None:
                metrics = {
                    key: float(value) for key, value in trainer.callback_metrics.items()
                }
            else:
                metrics = {
                    key: float(value)
                    for key, value in trainer.callback_metrics.items()
                    if key in self.metrics_to_log
                }
            wandb.log(metrics)

        def on_fit_end(self, trainer, pl_module):
            metric_val = trainer.callback_metrics.get("val_loss")
            if metric_val is None:
                return

            wandb.summary[f"best_{'val_loss'}"] = metric_val.item()  # keep in summary

    return WandbLogger()
