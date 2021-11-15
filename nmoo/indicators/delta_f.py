"""
ΔF performance indicator
"""
__docformat__ = "google"

from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
from pymoo.core.indicator import Indicator
from pymoo.core.problem import Problem

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
    """The ground problem."""

    _history: Dict[str, np.ndarray]
    """
    Contains all `X` values submitted to this indicator, and all denoised `F`
    values.
    """

    _n_evals: int
    """
    Number of evaluations of the ground problem that will be averaged in order
    to approximate the ground Pareto front.
    """

    def __init__(self, problem: WrappedProblem, n_evals: int = 1, **kwargs):
        super().__init__(zero_to_one=False, **kwargs)
        self._ground_problem = problem.ground_problem()
        self._history = {}
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
            out: Dict[str, Any] = {}
            self._ground_problem._evaluate(X, out, *args, **kwargs)
            fs.append(out["F"])
        af = np.mean(np.array(fs), axis=0)

        self._history["X"] = (
            np.append(self._history["X"], X, axis=0)
            if "X" in self._history
            else X
        )
        self._history["F"] = (
            np.append(self._history["F"], af, axis=0)
            if "F" in self._history
            else af
        )

        return np.mean(np.linalg.norm(F - af, axis=-1))

    def dump_history(self, path: Union[Path, str], compressed: bool = True):
        """
        Dumps the history into an NPZ archive.

        Args:
            path (Union[Path, str]): File path of the output archive.
            compressed (bool): Wether to compress the archive (defaults to
                `True`).
        """
        saver = np.savez_compressed if compressed else np.savez
        saver(path, **self._history)
