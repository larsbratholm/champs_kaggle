import math
import pandas as pd
import numpy as np

class CosineAnnealing:
    def __init__(self, step_start, step_end, lr_start, lr_end):
        self.step_start = step_start
        self.step_end = step_end
        self.lr_start = lr_start
        self.lr_end = lr_end

    def get(self, step):
        if step >= self.step_start and step <= self.step_end:
            lr = (
                self.lr_end + 
                0.5 * (self.lr_start - self.lr_end) *     
                (1 + math.cos((step - self.step_start) / (self.step_end - self.step_start) * math.pi))
            )
        else:
            lr = None

        return lr
   
    def plot(self, **kwargs):
        lrs = []
        index = []
        for step in range(self.step_start, self.step_end):
            lr = self.get(step)
            if lr is not None:
                lrs.append(self.get(step))
                index.append(step)
        lrs = pd.Series(lrs, index = index, name = 'lr')
        lrs.index.name = 'step'
        lrs.plot(**kwargs)
        
class LinearScheduler:
    def __init__(self, step_start, step_end, lr_start, lr_end):
        self.step_start = step_start
        self.step_end = step_end
        self.lr_start = lr_start
        self.lr_end = lr_end

    def get(self, step):
        if step >= self.step_start and step <= self.step_end:
            lr = self.lr_start + (self.lr_end - self.lr_start) * (step - self.step_start) / (self.step_end - self.step_start)
        else:
            lr = None

        return lr
   
    def plot(self, **kwargs):
        lrs = []
        index = []
        for step in range(self.step_start, self.step_end):
            lr = self.get(step)
            if lr is not None:
                lrs.append(self.get(step))
                index.append(step)
        lrs = pd.Series(lrs, index = index, name = 'lr')
        lrs.index.name = 'step'
        lrs.plot(**kwargs)

class MixedScheduler:
    def __init__(self, schedulers):
        self.schedulers = sorted(schedulers, key = lambda s : s.step_start)
        self.step_start = self.schedulers[0].step_start
        self.step_end = self.schedulers[-1].step_end
    
    def get(self, step):
        lr = None
        for scheduler in self.schedulers:
            scheduler_lr = scheduler.get(step)
    
            if scheduler_lr is not None:
                lr = scheduler_lr
        return lr
    
    def plot(self, **kwargs):
        lrs = []
        index = []
        for step in range(self.step_start, self.step_end):
            lr = self.get(step)
            if lr is not None:
                lrs.append(self.get(step))
                index.append(step)
        lrs = pd.Series(lrs, index = index, name = 'lr')
        lrs.index.name = 'step'
        lrs.plot(**kwargs)
        
class ExpScheduler:
    def __init__(self, step_start, step_end, lr_start, lr_end):
        self.step_start = step_start
        self.step_end = step_end
        self.lr_start = lr_start
        self.lr_end = lr_end

        self.factor_delta = self.lr_end / self.lr_start
        self.step_delta = self.step_end - self.step_start
        self.factor = np.exp(np.log(self.factor_delta) / self.step_delta)
        
        
    def get(self, step):
        if step >= self.step_start and step <= self.step_end:
            step_delta = step - self.step_start            
            lr = self.lr_start * self.factor ** step_delta
        else:
            lr = None

        return lr
   
    def plot(self, **kwargs):
        lrs = []
        index = []
        for step in range(self.step_start, self.step_end):
            lr = self.get(step)
            if lr is not None:
                lrs.append(self.get(step))
                index.append(step)
        lrs = pd.Series(lrs, index = index, name = 'lr')
        lrs.index.name = 'step'
        lrs.plot(**kwargs)