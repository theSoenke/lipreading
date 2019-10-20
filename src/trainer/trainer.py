import torch
from tqdm import tqdm


class Trainer():
    def __init__(self, seed=0, gpu_id=0, num_max_epochs=100, checkpoint_callback=None, early_stop_callback=None, logger=None):
        self.gpu_id = gpu_id
        self.num_max_epochs = num_max_epochs
        self.checkpoint_callback = checkpoint_callback
        self.early_stop_callback = early_stop_callback
        self.logger = logger

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device(f"cuda:{gpu_id}" if self.use_gpu else "cpu")
        self.current_epoch = 0

    def fit(self, model):
        self.model = model
        self.optimizer = self.model.configure_optimizers()
        self.model.to(self.device)
        self.model.train()
        dataloader, train_samples = model.train_dataloader()
        print(f"Training samples: {train_samples}")

        for epoch in range(self.num_max_epochs):
            self.current_epoch += 1
            with tqdm(total=len(dataloader)) as pbar:
                pbar.set_description(f"Epoch {epoch}")
                for batch in dataloader:
                    if self.use_gpu:
                        batch = self.transfer_batch_to_gpu(batch, self.gpu_id)
                    output = model.training_step(batch)
                    pbar.set_postfix(self.__process_logs(output))
                    if 'loss' in output:
                        output['loss'].backward()
                    model.optimizer_step(self.optimizer)
                    self.__log_metrics(output)
                    pbar.update(1)

            logs = self.validate(self.model)
            self.__log_metrics(logs)
            processed_logs = self.__process_logs(logs)
            if self.checkpoint_callback != None:
                self.checkpoint_callback.on_epoch_end(epoch, save_func=self.save_checkpoint, logs=processed_logs)

            if self.early_stop_callback != None:
                stop_training = self.early_stop_callback.on_epoch_end(epoch=epoch, logs=processed_logs)
                if stop_training:
                    break

    @torch.no_grad()
    def validate(self, model):
        model.to(self.device)
        model.eval()
        dataloader, val_samples = model.val_dataloader()
        print(f"Validation samples: {val_samples}")

        outputs = []
        with tqdm(total=len(dataloader)) as pbar:
            for batch in dataloader:
                pbar.set_description("Validation")
                if self.use_gpu:
                    batch = self.transfer_batch_to_gpu(batch, self.gpu_id)
                output = model.validation_step(batch)
                outputs.append(output)
                pbar.update(1)

        model.train()
        eval_results = self.model.validation_end(outputs)
        return eval_results

    def test(self, model):
        model.to(self.device)
        # TODO

    def __process_logs(self, logs):
        metrics = {}
        for key, value in logs.items():
            if key == 'log':
                continue
            if isinstance(value, torch.Tensor):
                metrics[key] = value.item()
            else:
                metrics[key] = value

        return metrics

    def __log_metrics(self, outputs):
        if self.logger != None and 'log' in outputs:
            processed_logs = self.__process_logs(outputs['log'])
            self.logger.log_metrics(processed_logs)

    def transfer_batch_to_gpu(self, batch, gpu_id):
        if callable(getattr(batch, 'cuda', None)):
            return batch.cuda(gpu_id)
        elif callable(getattr(batch, 'to', None)):
            return batch.to(torch.device('cuda', gpu_id))
        elif isinstance(batch, list):
            for i, x in enumerate(batch):
                batch[i] = self.transfer_batch_to_gpu(x, gpu_id)
            return batch
        elif isinstance(batch, tuple):
            batch = list(batch)
            for i, x in enumerate(batch):
                batch[i] = self.transfer_batch_to_gpu(x, gpu_id)
            return tuple(batch)
        elif isinstance(batch, dict):
            for k, v in batch.items():
                batch[k] = self.transfer_batch_to_gpu(v, gpu_id)

            return batch

        return batch

    def save_checkpoint(self, filepath):
        checkpoint = {
            'optimizer_state': self.optimizer.state_dict(),
            'state_dict': self.model.state_dict(),
        }
        torch.save(checkpoint, filepath)
