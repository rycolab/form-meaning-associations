
class TrainInfo:
    epoch_id = 0
    # running_loss = []
    best_loss = float('inf')
    best_epoch = 0

    def __init__(self, wait_epochs, pbar):
        self.wait_epochs = wait_epochs
        self.pbar = pbar

    @property
    def finish(self):
        if (self.epoch_id - self.best_epoch) >= self.wait_epochs:
            self.pbar.close()
            return True

        return False


    @property
    def max_epochs(self):
        return self.best_epoch + self.wait_epochs

    def is_best(self, val_loss, val_acc):
        self.epoch_id += 1

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_acc = val_acc
            self.best_epoch = self.epoch_id
            return True

        return False

    def update_info(self, train_loss, val_loss, val_acc):
        self.pbar.total = self.max_epochs
        self.pbar.update(1)
        self.pbar.set_description(
            '%d/%d: train_loss %.3f  val_loss: %.3f  acc: %.3f  best_loss: %.3f  acc: %.3f' %
            (self.epoch_id, self.best_epoch, train_loss, val_loss, val_acc, self.best_loss, self.best_acc))
