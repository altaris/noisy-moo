"""
Averaging: a naive method to reduce 0-mean noise.
"""
__docformat__ = "google"

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from joblib import Parallel, delayed
from loguru import logger as logging
from pymoo.core.problem import Problem

from nmoo.wrapped_problem import WrappedProblem


class ResampleAverage(WrappedProblem):
    """
    A resampler tries to denoise a noisy problem by evaluating a solution
    multiple times and averaging the outputs.
    """

    _cache_dir: Optional[Path]
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
        cache_dir: Optional[Path] = None,
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
            cache_dir (Optional[Path]): TLDR: don't touch that. If set, each
                evaluation of the problem will be cached so that if this
                wrapper's `_evaluate` is interrupted somehow, the data
                calculated so far won't be lost. The data will be saved in
                directory `cache_dir` (which will be created if needed) under
                `0.npz`, `1.npz`, etc... Warning: Use a different directory for
                each instance of `ResampleAverage` for which caching is
                enabled. Warning: Use a cache-enabled `ResampleAverage` for
                only one evaluation since in this case, all inputs passed to
                successive calls to `evaluate` are assumed to be the same. Yes
                this thing was introduced as a hotfix.
        """
        super().__init__(problem, name=name, **kwargs)
        self._cache_dir = cache_dir
        if n_evaluations <= 0:
            raise ValueError("The number of evaluations should be at least 1.")
        self._n_evaluations = n_evaluations
        self._n_jobs = n_jobs
        self._parallel = parallel

    def _evaluate(self, x, out, *args, **kwargs):
        def _f(x: np.ndarray, i: int, *args, **kwargs) -> dict:
            if self._cache_dir is not None:
                p = self._cache_dir / f"{i}.npz"
                if p.is_file():
                    logging.debug("Loading cached evaluation from '{}'", p)
                    return dict(np.load(p))
            tmp: dict = {}
            self._problem._evaluate(x, tmp, *args, **kwargs)
            if self._cache_dir is not None:
                p = self._cache_dir / f"{i}.npz"
                logging.debug("Saving evaluation to cache file '{}'", p)
                np.savez(p, **tmp)
            return tmp

        if self._cache_dir is not None and not os.path.isdir(self._cache_dir):
            os.makedirs(self._cache_dir)

        outs: List[dict] = []
        if self._parallel:
            executor = Parallel(n_jobs=self._n_jobs)
            outs = executor(
                delayed(_f)(x, i, *args, **kwargs)
                for i in range(self._n_evaluations)
            )
        else:
            for i in range(self._n_evaluations):
                outs.append(_f(x, i, *args, **kwargs))
        for k in outs[0]:  # By assumption, self._n_evaluations > 0
            # TODO: What if the type is not consistent across evaluations?
            if isinstance(outs[0][k], (int, float, np.ndarray)):
                out[k] = np.average([o[k] for o in outs], axis=0)
            else:
                # TODO: Deal with different non numeric outputs
                out[k] = outs[-1][k]
        self.add_to_history_x_out(x, out)
