"""
Random noises to apply to objective functions.
"""

import logging
from typing import Dict, Tuple

from pymoo.model.problem import Problem
import numpy as np

from .utils import *


Noise = ProblemWrapper
"""
An abstract noisy problem
"""


class GaussianNoise(Noise):
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
        self._problem._evaluate(x, out, *args, **kwargs)
        noises = dict()
        for k in self._parameters.keys():
            try:
                mean, stddev = self._parameters[k]
                noises[k] = np.random.normal(mean, stddev, out[k].shape)
                out[k] += noises[k]
            except KeyError:
                logging.error(
                    "Noise parameter key %s is not present objective function "
                    "objective function keys: %s.",
                    k,
                    str(list(out.keys())),
                )
        self.add_to_history(
            pd.concat(
                [x_out_to_df(x, out)]
                + [np2d_to_df(v, k + "_noise") for k, v in noises.items()],
                axis=1,
            )
        )
