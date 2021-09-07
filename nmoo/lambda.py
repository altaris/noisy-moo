"""
A wrapper that applies a given function to the wrapped problem's output.
"""
__docformat__ = "google"

from typing import Callable, Dict
import logging

from pymoo.model.problem import Problem
import numpy as np

from .wrapped_problem import WrappedProblem


class Lambda(WrappedProblem):
    """
    A wrapper that applies a given function to the wrapped problem's output.
    """

    _functions: Dict[str, Callable[[np.ndarray], np.ndarray]]
    """Functions to apply."""

    def __init__(
        self,
        problem: Problem,
        functions: Dict[str, Callable[[np.ndarray], np.ndarray]],
    ):
        """
        Constructor.

        Args:
            problem (:obj:`Problem`): A non-noisy pymoo problem.
            parameters (dict): Functions to apply, in the form of a dict
                mapping the name of an objective to a callable object. The set
                of keys should be a subset of the final `out` dictionary keys
                in the wrapped problem's `_evaluate` method.

        Example:
            Assume that the output of the wrapped problem has an `F`
            numerical objective. The following multiplies its value by 2:

                problem2 = nmoo.Lambda(
                    problem,
                    {
                        "F": lambda x: 2. * x,
                    },
                )
        """
        super().__init__(problem)
        self._functions = functions

    def _evaluate(self, x, out, *args, **kwargs):
        self._problem._evaluate(x, out, *args, **kwargs)
        for k in self._functions.keys():
            try:
                function = self._parameters[k]
                out[k] = function(out[k])
            except KeyError:
                logging.error(
                    "Callable key %s is not present in objective "
                    "function output keys. This objective will not be "
                    "modified. Objective function keys: %s. ",
                    k,
                    str(list(out.keys())),
                )
        self.add_to_history_x_out(x, out)