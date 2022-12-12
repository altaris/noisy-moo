"""
ΔF performance indicator, which measures how accurately a problem estimates the
true objective. To measure accuracy for Pareto individuals only, see
`nmoo.indicators.delta_f_pareto.DeltaFPareto`.
"""
__docformat__ = "google"

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from pymoo.core.indicator import Indicator
from pymoo.core.problem import Problem

from nmoo.wrapped_problem import WrappedProblem


class DeltaF(Indicator):
    """
    The ΔF performance indicator. Given
    * a `nmoo.wrapped_problem.WrappedProblem` with `g` its ground problem,
    * a population `X`,
    * and their image (under the objective function) array `F`,

    calculates the vectorized average Euclidean distance between `F` and `Y`,
    where `Y` is obtained by averaging `g(X)` a given number of times.

    You can use this PI in a benchmark as
    ```py
    Benchmark(
        ...
        performance_indicators=[..., "df", ...]
    )
    ```
    """

    _ground_problem: Problem
    """The ground problem."""

    _history: Dict[str, np.ndarray]
    """
    Contains all `X` values submitted to this indicator, and all denoised `F`
    values.
    """

    _history_path: Optional[Path]

    _n_evals: int
    """
    Number of evaluations of the ground problem that will be averaged in order
    to approximate the ground Pareto front.
    """

    def __init__(
        self,
        problem: WrappedProblem,
        n_evals: int = 1,
        history_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        super().__init__(zero_to_one=False, **kwargs)
        self._ground_problem = problem.ground_problem()
        self._history_path = (
            Path(history_path)
            if isinstance(history_path, str)
            else history_path
        )
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
        X, fs = args[0], []
        for _ in range(self._n_evals):
            out: Dict[str, Any] = {}
            self._ground_problem._evaluate(X, out, *args, **kwargs)
            fs.append(out["F"])
        self._history = {"X": X, "F": np.mean(np.array(fs), axis=0)}
        self.dump_history()
        return np.mean(np.linalg.norm(F - self._history["F"], axis=-1))

    def dump_history(self):
        """Dumps the history into an NPZ archive"""
        if isinstance(self._history_path, Path):
            np.savez_compressed(self._history_path, **self._history)
