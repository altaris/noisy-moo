"""
Averaging: a naive method to reduce 0-mean noise.
"""
__docformat__ = "google"

from typing import List

from pymoo.core.problem import Problem
import numpy as np

from nmoo.wrapped_problem import WrappedProblem


class ResampleAverage(WrappedProblem):
    """
    A resampler tries to denoise a noisy problem by evaluating a solution
    multiple times and averaging the outputs.
    """

    _n_evaluations: int

    def __init__(
        self,
        problem: Problem,
        n_evaluations: int = 5,
        *,
        name: str = "resample_avg",
        **kwargs,
    ):
        """
        Constructor.

        Args:
            problem (:obj:`Problem`): Noisy pymoo problem.
            n_evaluations (int): Number of times to evaluate the problem on
                each solution. Defaults to 5.
            name (str): An optional name for this problem. This will be used
                when creating history dump files. Defaults to
                `resample_avg`.
        """
        super().__init__(problem, name=name, **kwargs)

        if n_evaluations <= 0:
            raise ValueError("The number of evaluations should be at least 1.")
        self._n_evaluations = n_evaluations

    def _evaluate(self, x, out, *args, **kwargs):
        outs: List[dict] = []
        for _ in range(self._n_evaluations):
            outs.append({})
            self._problem._evaluate(x, outs[-1], *args, **kwargs)
        for k in outs[0]:  # By assumption, self._n_evaluations > 0
            # TODO: What if the type is not consistent across evaluations?
            if isinstance(outs[0][k], (int, float, np.ndarray)):
                out[k] = np.average([o[k] for o in outs], axis=0)
            else:
                # TODO: Deal with different non numeric outputs
                out[k] = outs[-1][k]
        self.add_to_history_x_out(x, out)
