from typing import Tuple, Optional, List

import glpk
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
    x_ = Q1.__matmul__(np.linalg.inv(R.T).__matmul__(b))
    return x_, Q2


def least_p(A: np.ndarray, b: np.ndarray, p: float = 2.0) -> np.ndarray:
    '''
    Minimizer of ||Ax+b||_p^p
    '''
    if len(A.shape) != 2 or len(b.shape) != 1:
        raise Exception("A must be 2D, b must be 1D")

    m, n = A.shape

    def f(x: np.ndarray) -> np.ndarray:
        return A.__matmul__(x).__add__(b)

    def objective(x: np.ndarray) -> float:
        return float(np.sum(np.abs(f(x))))

    def gradient(x: np.ndarray) -> np.ndarray:
        fx = f(x)
        return A.T.__matmul__(np.sign(fx).__mul__(np.abs(fx).__pow__(p - 1)).__mul__(p))

    def hessian(x: np.ndarray) -> np.ndarray:
        return A.T.__matmul__(A).__mul__(p * (p-1))

    x0 = np.zeros(shape=(n,))
    solution = sp.optimize.minimize(objective, x0, method="L-BFGS-B", jac=gradient, hess=hessian)
    return solution.x

def glpk_linprog(
        c: np.ndarray,
        A: Optional[np.ndarray] = None,
        b_lb: Optional[np.ndarray] = None,
        b_ub: Optional[np.ndarray] = None,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
) -> np.ndarray:
    '''
    Solving linear program as follow:
        minimize c^T x
        given b_lb <= A x <= b_ub and A_eq x = b_eq
    bounds is a list of pair (lb, ub) such as lb[i] < x[i] < ub[i]
    '''
    num_variables = c.shape[0]
    if A is None:
        A = np.empty(shape=(0, num_variables))
        b_lb = np.empty(shape=(0, ))
        b_ub = np.empty(shape=(0, ))
    else:
        if b_lb is None:
            b_lb = [None for _ in range(num_variables)]
        if b_ub is None:
            b_ub = [None for _ in range(num_variables)]
    if A_eq is None:
        A_eq = np.empty(shape=(0, num_variables))
        b_eq = np.empty(shape=(0, ))
    if bounds is None:
        bounds = [(None, None) for _ in range(num_variables)]


    num_ineq_constraints = A.shape[0]
    num_eq_constraints = A_eq.shape[0]

    lp = glpk.LPX()
    lp.obj.maximize = False
    lp.rows.add(num_ineq_constraints + num_eq_constraints)
    for i_b in range(num_ineq_constraints):
        lp.rows[i_b].bounds = b_lb[i_b], b_ub[i_b]
    for i_eq in range(num_eq_constraints):
        lp.rows[i_eq + num_ineq_constraints].bounds = b_eq[i_eq]

    lp.cols.add(num_variables)
    for i in range(num_variables):
        lp.cols[i].bounds = bounds[i][0], bounds[i][1]

    lp.obj[:] = list(c)
    lp.matrix = list(np.vstack((A, A_eq)).flatten(order="C"))

    lp.simplex()

    x = np.array([col.primal for col in lp.cols])
    return x
