from typing import Tuple

import numpy as np


def linear_subspace(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Solution of Ax = b:
    x = x_ + Q2 z where z is an arbitrary vector
    '''
    # https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf (page 682)
    # https://math.stackexchange.com/questions/1942211/does-negative-transpose-sign-mean-inverse-of-a-transposed-matrix-or-transpose-of
    p, n = A.shape
    Q, R = np.linalg.qr(A.T, mode="complete")
    Q1, Q2 = Q[:, 0:p], Q[:, p:n]
    R = R[0:p, :]
    x_ = Q1.__matmul__(np.linalg.inv(R.T).__matmul__(b))
    return x_, Q2
