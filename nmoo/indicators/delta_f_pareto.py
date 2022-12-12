"""
ΔF-Pareto performance indicator, which measures how accurately a problem
estimates the true objective **of pareto individuals**, unlike
`nmoo.indicators.delta_f.DeltaF` which considers **all** individuals.
"""
__docformat__ = "google"

from typing import Any, Dict

import numpy as np

from nmoo.utils.population import pareto_frontier_mask
from .delta_f import DeltaF


class DeltaFPareto(DeltaF):
    """
    The ΔF-Pareto performance indicator. Given
    * a `nmoo.wrapped_problem.WrappedProblem` with `g` its ground problem,
    * a population `X`,
    * and their image (under the objective function) array `F`,

    calculates the vectorized average Euclidean distance between `F'` and `Y`,
    where `F'` is the subarray corresponding to Pareto individuals in `X`, and
    where `Y` is obtained by averaging `g(X)` a given number of times.

    You can use this PI in a benchmark as
    ```py
    Benchmark(
        ...
        performance_indicators=[..., "dfp", ...]
    )
    ```
    """

    # pylint: disable=duplicate-code
    def _do(self, F: np.ndarray, *args, **kwargs) -> float:
        """
        Calculates ΔF. See class documentation.
        """
        if not args:
            raise ValueError(
                "Need a second argument (namely the X array) when calling "
                "DeltaF.do"
            )
        pfm = pareto_frontier_mask(F)
        X, F, fs = args[0][pfm], F[pfm], []
        for _ in range(self._n_evals):
            out: Dict[str, Any] = {}
            self._ground_problem._evaluate(X, out, *args, **kwargs)
            fs.append(out["F"])
        self._history = {"X": X, "F": np.mean(np.array(fs), axis=0)}
        self.dump_history()
        return np.mean(np.linalg.norm(F - self._history["F"], axis=-1))
