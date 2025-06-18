from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @abstractmethod
    def train(self, train_loader, optimizer, **kwargs):
        pass
    
    def run_training(self, train_loader, optimizer, num_epochs, validation_loader=None, 
                     test_loader=None, normal_class=None, scheduler=None, early_stopper=None, logger=None, log_every=1):
        if early_stopper is not None:
            early_stopper = early_stopper(patience=10, use_train_loss=not self.semi)
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train(train_loader, optimizer)
            if scheduler:
                scheduler.step()
                
            info_train = f'Epoch {epoch:3d}, Train Loss {train_loss:.4f}'

            if epoch % log_every == 0:
                if validation_loader:
                    val_roc, val_pr = self.evaluate(validation_loader, normal_class)
                    info_val = f'VAL ROC: {val_roc:.4f}, VAL PR: {val_pr:.4f}'
                    if logger:
                        logger.log(info_train + '   ' + info_val)

            if early_stopper and early_stopper.stop(epoch, train_loss, val_roc if validation_loader else None):
                break
        if test_loader:
            test_roc, test_pr = self.evaluate(test_loader, normal_class)
            info_test = f'Test ROC: {test_roc:.4f}, Test PR: {test_pr:.4f}'
            if logger:
                logger.log(info_test)
        return test_roc if test_loader else None, test_pr if test_loader else None

    @abstractmethod
    def evaluate(self, data_loader, normal_class=0, **kwargs):
        pass