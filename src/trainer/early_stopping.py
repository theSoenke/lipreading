import numpy as np


class EarlyStopping():
    def __init__(self, monitor='val_loss', min_delta=0.0, patience=0, mode='max'):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

        self.on_train_begin()

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        stop_training = False
        if current is None:
            print(f"Early stopping metric `{self.monitor}` is not available. Available metrics: {','.join(list(logs.keys()))}", RuntimeWarning)
            stop_training = True
            return stop_training

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                stop_training = True
                self.on_train_end()

        return stop_training

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"Epoch {self.stopped_epoch + 1:05d}: early stopping")
