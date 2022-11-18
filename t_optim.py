'''A wrapper class for scheduled optimizer '''
import numpy as np

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer,epoch,lr=0.1):
        self._optimizer = optimizer
        self.epoch = epoch
        self.lr = lr


    def step_lr(self):
        "Step with the inner optimizer"
        self._optimizer.step()

    def update_lr(self,epoch):
        "Step with the inner optimizer"
        self._update_learning_rate(epoch)

    def _update_learning_rate(self,epoch):
        ''' Learning rate scheduling '''
        if epoch<30:
            for param_group in self._optimizer.param_groups:
                # print('before', param_group['lr'])
                self.lr = param_group['lr'] * 1.4
                # print('after', param_group['lr'])
                param_group['lr'] = self.lr
        else:
            for param_group in self._optimizer.param_groups:
                # print('before', param_group['lr'])
                self.lr = param_group['lr'] * 0.6
                # print('after', param_group['lr'])
                param_group['lr'] = self.lr
    def get_lr(self):
        return self.lr