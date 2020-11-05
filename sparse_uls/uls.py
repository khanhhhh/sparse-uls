import numpy as np

from sparse_uls.util import linear_subspace, least_p


def solve(A: np.ndarray, b: np.ndarray, p: float = 1.0) -> np.ndarray:
    '''
    Minimizer of ||x||_p^p
    Given Ax=b
    '''
    if len(A.shape) != 2 or len(b.shape) != 1:
        raise Exception("A must be 2D, b must be 1D")

    x_, Q2 = linear_subspace(A, b)
    z = least_p(Q2, x_, p)
    x = Q2.__matmul__(z).__add__(x_)
    return x