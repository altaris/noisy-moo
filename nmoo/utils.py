"""
Various utilities.
"""

from pymoo.model.problem import Problem
import numpy as np


class ProblemWrapper(Problem):
    """
    A noise class is a pymoo problem wrapping another (non noisy) problem.
    """

    _problem: Problem

    def __init__(self, problem: Problem):
        """
        Constructor.

        Args:
            problem (:obj:`Problem`): A non-noisy pymoo problem.
        """
        super().__init__(
            n_var=problem.n_var,
            n_obj=problem.n_obj,
            n_constr=problem.n_constr,
            xl=problem.xl,
            xu=problem.xu,
            type_var=problem.type_var,
            evaluation_of=problem.evaluation_of,
            replace_nan_values_of=problem.replace_nan_values_of,
            parallelization=problem.parallelization,
            elementwise_evaluation=problem.elementwise_evaluation,
            exclude_from_serialization=problem.exclude_from_serialization,
            callback=problem.callback,
        )
        self._problem = problem

    def _evaluate(self, x, out, *args, **kwargs):
        return self._problem._evaluate(x, out, *args, **kwargs)
