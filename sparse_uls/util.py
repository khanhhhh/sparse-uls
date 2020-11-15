from typing import Tuple

import numpy as np
import scipy as sp
import scipy.optimize

def linear_subspace(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Solution of Ax = b:
    x = x_ + Q2 z where z is an arbitrary vector
    '''
    # https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf (page 682)
    # https://math.stackexchange.com/questions/1942211/does-negative-transpose-sign-mean-inverse-of-a-transposed-matrix-or-transpose-of
    if len(A.shape) != 2 or len(b.shape) != 1:
        raise Exception("A must be 2D, b must be 1D")
    p, n = A.shape
    Q, R = sp.linalg.qr(A.T, mode="full")
    Q1, Q2 = Q[:, 0:p], Q[:, p:n]
    R = R[0:p, :]
    x_ = Q1 @ (np.linalg.inv(R.T) @ b)
    return x_, Q2


def least_p(A: np.ndarray, b: np.ndarray, p: float = 2.0) -> np.ndarray:
    '''
    Minimizer of ||Ax+b||_p^p
    '''
    if len(A.shape) != 2 or len(b.shape) != 1:
        raise Exception("A must be 2D, b must be 1D")

    m, n = A.shape

    def f(x: np.ndarray) -> np.ndarray:
        return A @ x + b

    def objective(x: np.ndarray) -> float:
        return float(np.sum(np.abs(f(x))))

    def gradient(x: np.ndarray) -> np.ndarray:
        fx = f(x)
        return A.T @ (p * np.sign(fx) * np.abs(fx) ** (p-1))

    def hessian(x: np.ndarray) -> np.ndarray:
        return A.T.__matmul__(A).__mul__(p * (p-1))

    x0 = np.zeros(shape=(n,))
    solution = sp.optimize.minimize(objective, x0, method="L-BFGS-B", jac=gradient, hess=hessian)
    return solution.x

