"""
Random noises to apply to objective functions.
"""
__docformat__ = "google"

import logging
from typing import Dict, Tuple

from pymoo.model.problem import Problem
import numpy as np

from nmoo.utils import ProblemWrapper


class GaussianNoise(ProblemWrapper):
    """
    A Gaussian noisy problem.
    """

    _parameters: Dict[str, Tuple[float, float]]

    def __init__(
        self,
        problem: Problem,
        parameters: Dict[str, Tuple[float, float]],
    ):
        """
        Constructor.

        Args:
            problem (:obj:`Problem`): A non-noisy pymoo problem.
            parameters (dict): Gaussian noise parameters, in the form of a dict
                mapping the name of an objective to a float pair (mean, stddev).
                The set of keys should exactly match that of the final out
                dictionary in the wrapped problem's `_evaluate` method.

        Example:
            The following creates a new problem by adding a `N(0, 2)` noise on
            the `F` component of the objective function of `problem`, and an
            `N(0, 0.25)` noise on the `G` component::

                noisy_problem = nmoo.GaussianNoise(
                    problem,
                    {
                        "F": (0., 2.),
                        "G": (0., .25),
                    },
                )

        See also:
            https://pymoo.org/getting_started.html#By-Class
        """
        super().__init__(problem)
        self._parameters = parameters

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Calls the wrapped problems's `_evaluate` method and adds a Gaussian
        noise. Adds the input (`x`), the noisy output, and the noise values to
        history.

        Example:

            If the wrapped problem's output dict looks like::

                {
                    "A": (n, m) np.ndarray,
                    "B": not an np.ndarray
                }

            then the history will look like this::

                {
                    "A": an np.ndarray,
                    "A_noise": an np.ndarray of the same dimension
                }

        """
        self._problem._evaluate(x, out, *args, **kwargs)
        noises: Dict[str, np.ndarray] = dict()
        for k in self._parameters.keys():
            try:
                mean, stddev = self._parameters[k]
                noises[k] = np.random.normal(mean, stddev, out[k].shape)
                out[k] += noises[k]
            except KeyError:
                logging.error(
                    "Noise parameter key %s is not present in objective "
                    "function output keys. No noise will be applied. "
                    "Objective function keys: %s. ",
                    k,
                    str(list(out.keys())),
                )
        self.add_to_history_x_out(
            x, out, **{k + "_noise": v for k, v in noises.items()}
        )
