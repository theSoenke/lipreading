from pytorch_lightning.logging import LightningLoggerBase, rank_zero_only

import wandb


class WandbLogger(LightningLoggerBase):
    def __init__(self, project, model):
        super().__init__()
        wandb.init(project=project)
        wandb.watch(model)

    @rank_zero_only
    def log_hyperparams(self, params):
        wandb.config.update(params)

    @rank_zero_only
    def log_metrics(self, metrics, step_num):
        wandb.log(metrics)

    def log(self, key, value):
        wandb.config[key] = value

    def save(self):
        pass

    @rank_zero_only
    def finalize(self, status):
        pass
