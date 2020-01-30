import numpy as np


def cosinus(x0, x1, x2):
    e0 = (x0-x1)
    e1 = (x2-x1)
    e0 = (e0 / np.linalg.norm(e0))
    e1 = (e1 / np.linalg.norm(e1))
    cosinus = (e0 * e1).sum(axis=1)
    return cosinus


def dihedral(x0, x1, x2, x3):

    b0 = -1.0 * (x1 - x0)
    b1 = x2 - x1
    b2 = x3 - x2

    b0xb1 = np.cross(b0, b1)
    b1xb2 = np.cross(b2, b1)

    b0xb1_x_b1xb2 = np.cross(b0xb1, b1xb2)

    y = (b0xb1_x_b1xb2 * b1).sum(axis=1) * (1.0/np.linalg.norm(b1))
    x = (b0xb1 * b1xb2).sum(axis=1)
    
    grad = np.arctan2(y, x)
    return grad

