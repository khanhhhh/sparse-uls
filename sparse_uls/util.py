from typing import Tuple, Optional, Callable, Union, Iterator, Dict

import numpy as np
import torch


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


def lbfgs_optimizer(
        lr: float = 1,
        max_iter: int = 20,
        max_eval: Optional[int] = None,
        tolerance_grad: float = 1e-07,
        tolerance_change: float = 1e-09,
        history_size: int = 100,
        line_search_fn: Optional[str] = None,
) -> Callable[[Union[Iterator[torch.Tensor], Iterator[Dict]]], torch.optim.Optimizer]:
    def optimizer(params: Union[Iterator[torch.Tensor], Iterator[Dict]]) -> torch.optim.Optimizer:
        return torch.optim.LBFGS(
            params=params,
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,
        )

    return optimizer
