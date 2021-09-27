"""
ΔF performance indicator
"""
__docformat__ = "google"

from typing import Any, Dict

from pymoo.core.indicator import Indicator
from pymoo.core.problem import Problem
import numpy as np

from nmoo.wrapped_problem import WrappedProblem


class DeltaF(Indicator):
    """
    The ΔF performance indicator. Given
    * a `WrappedProblem` `p`, with `g` its ground problem,
    * a Pareto set `F`,
    * and a population `X`,

    calculates the vectorized average Euclidean distance between `F` and `Y`,
    where `Y` is obtained by averaging `g(X)` a given number of times.
    """

    _ground_problem: Problem
    """
    The ground problem.
    """

    _n_evals: int
    """
    Number of evaluations of the ground problem that will be averaged in order
    to approximate the ground Pareto front.
    """

    def __init__(self, problem: WrappedProblem, n_evals: int = 1, **kwargs):
        super().__init__(zero_to_one=False, **kwargs)
        self._ground_problem = problem.ground_problem()
        if n_evals < 1:
            raise ValueError("n_evals must be >= 1")
        self._n_evals = n_evals

    def _do(self, F: np.ndarray, *args, **kwargs) -> float:
        """
        Calculates ΔF. See class documentation.
        """
        if not args:
            raise ValueError(
                "Need a second argument (namely the X array) when calling "
                "DeltaF.do"
            )
        X = args[0]
        fs = []
        for _ in range(self._n_evals):
            out: Dict[str, Any] = dict()
            self._ground_problem._evaluate(X, out, *args, **kwargs)
            fs.append(out["F"])
        af = np.mean(np.array(fs), axis=0)
        return np.mean(np.linalg.norm(F - af, axis=-1))
