from pytorch_lightning.callbacks import Callback

import wandb


def wandb_logger():
    """Lightweight W&B logging callback for Lightning models."""

    class WandbLogger(Callback):
        def __init__(self, metrics_to_log=None):
            self.metrics_to_log = metrics_to_log or ["train_loss", "val_loss"]

        def on_train_start(self, trainer, pl_module):
            wandb.watch(pl_module)

        def _log_selected(self, trainer):
            if self.metrics_to_log is None:
                metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
            else:
                metrics = {
                    k: float(v)
                    for k, v in trainer.callback_metrics.items()
                    if k in self.metrics_to_log
                }
            if metrics:
                wandb.log(metrics)

        def on_train_epoch_end(self, trainer, pl_module):
            self._log_selected(trainer)

        def on_validation_epoch_end(self, trainer, pl_module):
            self._log_selected(trainer)

    return WandbLogger()
