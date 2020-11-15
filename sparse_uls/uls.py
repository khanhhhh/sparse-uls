from enum import Enum

import numpy as np
import scipy as sp
import scipy.optimize
from sparse_uls.lp import scipy_linprog, octave_linprog, glpk_linprog


def solve_homopoly(A: np.ndarray, b: np.ndarray, p: float = 2.0) -> np.ndarray:
    '''
    Minimizer of ||x||_p^p
    Given Ax=b
    '''
    if len(A.shape) != 2 or len(b.shape) != 1:
        raise Exception("A must be 2D, b must be 1D")

    m, n = A.shape

    if not (m < n):
        raise Exception("System must be underdetermined (m < n)")

    def objective(x: np.ndarray) -> np.ndarray:
        return np.sum((A.__matmul__(x).__sub__(b)).__pow__(2)).__add__(np.sum(np.abs(x).__pow__(p)))

    def gradient(x: np.ndarray) -> np.ndarray:
        y = A.__matmul__(x).__sub__(b)
        return A.T.__matmul__(y).__mul__(2).__add__(np.sign(x).__mul__(np.abs(x).__pow__(p-1)).__mul__(p))

    x0 = np.zeros(shape=(n,))
    solution = sp.optimize.minimize(objective, x0, method="L-BFGS-B", jac=gradient)
    return solution.x


def solve(A: np.ndarray, b: np.ndarray, p: float = 1.0) -> np.ndarray:
    '''
    Minimizer of ||x||_p^p
    Given Ax=b
    '''
    if len(A.shape) != 2 or len(b.shape) != 1:
        raise Exception("A must be 2D, b must be 1D")

    m, n = A.shape

    if not (m < n):
        raise Exception("System must be underdetermined (m < n)")

    if p == 1:
        return solve_1(A, b)

    return solve_homopoly(A, b, p)

class LPmethod(Enum):
    GLPK = 0
    OCTAVE = 1
    SCIPY = 2


def solve_1(A: np.ndarray, b: np.ndarray, method: LPmethod = LPmethod.GLPK) -> np.ndarray:
    '''
    Minimizer of ||Ax+b||_1 using linear programming
    '''
    if len(A.shape) != 2 or len(b.shape) != 1:
        raise Exception("A must be 2D, b must be 1D")

    m, n = A.shape

    if not (m < n):
        raise Exception("System must be underdetermined (m < n)")

    A_ = np.empty(shape=(2 * n, 2 * n))
    A_[0:n, 0:n] = +np.identity(n)
    A_[n:2 * n, 0:n] = -np.identity(n)
    A_[0:n, n:2 * n] = -np.identity(n)
    A_[n:2 * n, n:2 * n] = -np.identity(n)
    b_ub = np.zeros(shape=(2 * n))

    c = np.empty(shape=(2 * n))
    c[0:n] = 0
    c[n:2 * n] = 1

    A_eq = np.empty(shape=(m, 2 * n))
    A_eq[:, 0:n] = A
    A_eq[:, n:2 * n] = 0
    b_eq = b

    if method == LPmethod.GLPK:
        x1 = glpk_linprog(
            c=c,
            A=A_,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
        )
        return x1[0:n]

    if method == LPmethod.OCTAVE:
        x1 = octave_linprog(
            c=c,
            A_ub=A_,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
        )
        return x1[0:n]

    if method == LPmethod.SCIPY:
        x1 = scipy_linprog(
            c=c,
            A_ub=A_,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=[(None, None) for _ in range(2 * n)],
        )
        return x1[0:n]
