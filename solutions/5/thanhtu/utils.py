import torch
import torch.nn.functional as F
import numpy as np
from constants import *

def compute_kaggle_metric(predict, coupling_value, coupling_type):

    mae     = [None]*NUM_COUPLING_TYPE
    log_mae = [None]*NUM_COUPLING_TYPE
    diff = np.fabs(predict-coupling_value)
    for t in range(NUM_COUPLING_TYPE):
        index = np.where(coupling_type==t)[0]
        if len(index)>0:
            m = diff[index].mean()
            log_m = np.log(m+1e-8)

            mae[t] = m
            log_mae[t] = log_m
        else:
            pass

    return mae, log_mae
