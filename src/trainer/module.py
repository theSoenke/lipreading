from torch import nn, optim


class Module(nn.Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def training_step(self, batch):
        """
        return loss(optimal), dict with metrics to display
        """
        raise NotImplementedError

    def validation_step(self, batch):
        raise NotImplementedError

    def test_step(self, batch):
        raise NotImplementedError

    def validation_end(self, outputs):
        raise NotImplementedError

    def test_end(self, outputs):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def optimizer_step(self, optimizer):
        optimizer.step()
        optimizer.zero_grad()

    def train_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def summarize(self, mode):
        raise NotImplementedError

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
