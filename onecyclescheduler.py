from keras.callbacks import Callback
from keras import backend as K
import numpy as np


class OneCycleScheduler(Callback):
    """
    This callback implements the fast.ai variant of the one cycle schedule, 
    introduced in [https://arxiv.org/pdf/1803.09820.pdf]

    Arguments:
        epochs: total number of training epochs
        max_lr: maximum allowed learning rate
        steps_per_epoch: number of iterations in one epoch
        moms: a tuple consisting of (min_mom, max_mom)
        div_factor: sets min lr as min_lr = max_lr / div_factor
        start_pct: cycle length of each phase
    """

    def __init__(self, epochs, max_lr, steps_per_epoch, moms = (0.95, 0.85), div_factor = 25, start_pct = 0.3):
        # initialize the instance variables
        self.max_lr = max_lr
        self.moms = moms
        self.div_factor = div_factor
        self.st1_epochs = int(np.floor(epochs * start_pct))
        self.st2_epochs = epochs - self.st1_epochs
        self.st1_steps = self.st1_epochs * steps_per_epoch
        self.st2_steps = self.st2_epochs * steps_per_epoch
        self.history = {"lrs": [], "moms": []}

    def __annealing_cos(self, start, end, pct):
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."

        cos_out = np.cos(np.pi * pct) + 1
        return end + (start - end) / 2 * cos_out

    def on_train_begin(self, logs = None):
        # initialize the necessary variables
        self.steps_so_far = 0

    def on_batch_begin(self, batch, logs = None):
        # increment the step count
        self.steps_so_far += 1

        # check to determine the training phase
        if self.steps_so_far <= self.st1_steps:
            # calculate the new learning rate
            new_lr = self.__annealing_cos(self.max_lr / self.div_factor,
                                          self.max_lr,
                                          self.steps_so_far / self.st1_steps)

            # calculate the new momentum
            new_mom = self.__annealing_cos(self.moms[0],
                                           self.moms[1],
                                           self.steps_so_far / self.st1_steps)

            # set the new learning rate and momentum
            K.set_value(self.model.optimizer.lr, new_lr)
            K.set_value(self.model.optimizer.momentum, new_mom)

        else:
            # calculate the new learning rate
            new_lr = self.__annealing_cos(self.max_lr,
                                          self.max_lr / self.div_factor,
                                          (self.steps_so_far - self.st1_steps) / self.st2_steps)

            # calculate the new momentum
            new_mom = self.__annealing_cos(self.moms[1],
                                           self.moms[0],
                                           (self.steps_so_far - self.st1_steps) / self.st2_steps)

            # set the new learning rate and momentum
            K.set_value(self.model.optimizer.lr, new_lr)
            K.set_value(self.model.optimizer.momentum, new_mom)

        # update the history attribute
        self.history["lrs"].append(new_lr)
        self.history["moms"].append(new_mom)
