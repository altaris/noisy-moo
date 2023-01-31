"""
Averaging: a naive method to reduce 0-mean noise.
"""
__docformat__ = "google"

from typing import List
from joblib import delayed, Parallel

from pymoo.core.problem import Problem
import numpy as np

from nmoo.wrapped_problem import WrappedProblem


class ResampleAverage(WrappedProblem):
    """
    A resampler tries to denoise a noisy problem by evaluating a solution
    multiple times and averaging the outputs.
    """

    _n_evaluations: int
    _n_jobs: int
    _parallel: bool

    def __init__(
        self,
        problem: Problem,
        n_evaluations: int = 5,
        *,
        name: str = "resample_avg",
        parallel: bool = False,
        n_jobs: int = -1,
        **kwargs,
    ):
        """
        Constructor.

        Args:
            problem (pymoo `Problem`): Noisy pymoo problem (or
                `nmoo.wrapped_problem.WrappedProblem`).
            n_evaluations (int): Number of times to evaluate the problem on
                each solution. Defaults to 5.
            name (str): An optional name for this problem. This will be used
                when creating history dump files. Defaults to `resample_avg`.
            parallel (bool): If set to `True`, the wrapper's `_evaluate` will
                resample the wrapped problem in parallel. Defaults to `True`.
            n_jobs (int): If `parallel` is set to `True`, this specifies the
                number of joblib jobs to use when resampling the wrapped
                problem.
        """
        super().__init__(problem, name=name, **kwargs)

        if n_evaluations <= 0:
            raise ValueError("The number of evaluations should be at least 1.")
        self._n_evaluations = n_evaluations
        self._n_jobs = n_jobs
        self._parallel = parallel

    def _evaluate(self, x, out, *args, **kwargs):
        def _f(x: np.ndarray, *args, **kwargs) -> dict:
            tmp: dict = {}
            self._problem._evaluate(x, tmp, *args, **kwargs)
            return tmp

        outs: List[dict] = []
        if self._parallel:
            executor = Parallel(n_jobs=self._n_jobs)
            outs = executor(
                delayed(_f)(x, *args, **kwargs)
                for _ in range(self._n_evaluations)
            )
        else:
            for _ in range(self._n_evaluations):
                outs.append(_f(x, *args, **kwargs))
        for k in outs[0]:  # By assumption, self._n_evaluations > 0
            # TODO: What if the type is not consistent across evaluations?
            if isinstance(outs[0][k], (int, float, np.ndarray)):
                out[k] = np.average([o[k] for o in outs], axis=0)
            else:
                # TODO: Deal with different non numeric outputs
                out[k] = outs[-1][k]
        self.add_to_history_x_out(x, out)
