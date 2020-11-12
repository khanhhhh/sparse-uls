from typing import Optional, List, Tuple

import glpk
import numpy as np
import scipy as sp
import scipy.optimize
from oct2py import octave


def scipy_linprog(
        c: np.ndarray,
        A_ub: Optional[np.ndarray] = None,
        b_ub: Optional[np.ndarray] = None,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
) -> np.ndarray:
    '''
    interior-point
    '''
    solution = sp.optimize.linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
    )
    return solution.x


def octave_linprog(
        c: np.ndarray,
        A_ub: Optional[np.ndarray] = None,
        b_ub: Optional[np.ndarray] = None,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
) -> np.ndarray:
    '''
    simplex
    '''
    octave.eval("pkg load optim")
    c_ = c.reshape((c.shape[0], 1))
    b_ub_ = b_ub.reshape((b_ub.shape[0], 1))
    b_eq_ = b_eq.reshape((b_eq.shape[0], 1))
    x1_ = octave.linprog(c_, A_ub, b_ub_, A_eq, b_eq_)
    x1 = x1_.reshape((x1_.shape[0],))
    return x1

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
    simplex
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

