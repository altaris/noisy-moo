"""
Averaging: a naive method to reduce 0-mean noise.
"""
__docformat__ = "google"

from typing import List

from pymoo.model.problem import Problem
import numpy as np

from nmoo.utils import ProblemWrapper


class Average(ProblemWrapper):
    """
    A resampler tries to denoise a noisy problem by evaluating a solution
    multiple times and averaging the outputs.
    """

    _n_evaluations: int

    def __init__(self, problem: Problem, n_evaluations: int = 5):
        """
        Constructor.

        Args:
            problem (:obj:`Problem`): Noisy pymoo problem.
            n_evaluations (int): Number of times to evaluate the problem on
                each solution. Defaults to 5.
        """
        super().__init__(problem)

        if n_evaluations <= 0:
            raise ValueError("The number of evaluations should be at least 1.")
        self._n_evaluations = n_evaluations

    def _evaluate(self, x, out, *args, **kwargs):
        outs: List[dict] = []
        for _ in range(self._n_evaluations):
            outs.append(dict())
            self._problem._evaluate(x, outs[-1], *args, **kwargs)
        for k in outs[0]:  # By assumption, self._n_evaluations > 0
            # TODO: What if the type is not consistent across evaluations?
            if isinstance(outs[0][k], (int, float, np.ndarray)):
                out[k] = np.average([o[k] for o in outs], axis=0)
            else:
                # TODO: Deal with different non numeric outputs
                out[k] = outs[-1][k]
        self.add_to_history_x_out(x, out)
