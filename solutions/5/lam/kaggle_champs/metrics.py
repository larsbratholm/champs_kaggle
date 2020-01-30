from . import constants
import torch
import numpy as np


class MeanLogGroupMAE(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum_absolute_err = np.array([0 for _ in range(len(constants.TYPES_LIST))], dtype='float32')
        self.count = np.array([0 for _ in range(len(constants.TYPES_LIST))])

    def update(self, pred, target, types):
        for i in range(8):
            self.sum_absolute_err[i] += ((pred - target).abs()
                                         * (types == i).to(torch.float)).sum().item()
            self.count[i] += (types == i).sum().item()

    def compute(self):
        log_mae = []
        for i in range(8):
            if self.count[i] != 0:
                log_mae.append(
                    np.log(max(self.sum_absolute_err[i] / self.count[i], 1e-9))
                )
        return np.mean(log_mae)

    def compute_individuals(self):
        mae = self.sum_absolute_err / np.maximum(self.count, 1e-3)
        logmae = np.log(np.maximum(mae, 1e-9))
        return dict([*zip(constants.TYPES_LIST, logmae)])


class AverageMetric(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, value, count):
        self.sum += value
        self.count += count

    def compute(self):
        return self.sum / self.count


m = MeanLogGroupMAE()
for _ in range(10):
    m.update(
        torch.zeros((8,1), dtype=torch.float),
        torch.zeros((8,1), dtype=torch.float),
        torch.arange(8).to(torch.float)
    )
assert round(m.compute(), 3) == -20.723