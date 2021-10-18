"""
Random noises to apply to objective functions.
"""
__docformat__ = "google"

import logging
from typing import Dict, Tuple

from pymoo.core.problem import Problem
import numpy as np

from nmoo.wrapped_problem import WrappedProblem


class GaussianNoise(WrappedProblem):
    """
    A Gaussian noisy problem.
    """

    _generator: np.random.Generator
    """Random number generator."""

    _parameters: Dict[str, Tuple[np.ndarray, np.ndarray]]
    """
    Noise parameters. Each entry is a tuple containing the noise's mean vector
    and covariance matrix.
    """

    def __init__(
        self,
        problem: Problem,
        parameters: Dict[str, Tuple[np.ndarray, np.ndarray]],
        *,
        name: str = "gaussian_noise",
    ):
        """
        Constructor.

        Args:
            name (str): An optional name for this problem. This will be used
                when creating history dump files. Defaults to
                `gaussian_noise`.
            problem (:obj:`Problem`): A non-noisy pymoo problem.
            parameters (dict): Gaussian noise parameters, in the form of a dict
                mapping the name of an objective to a numpy array pair
                (mean, covariance matrix). The set of keys should be a subset
                of the final `out` dictionary keys in the wrapped problem's
                `_evaluate` method.

        Example:
            Assume that the output of the wrapped problem has an `F` and a `G`
            numerical component, both of wich are 2-dimensional. The following
            creates a new problem by adding a `N(0, .2)` noise on all
            components of `F` (without any covariance), and a 0-mean
            1-dimensional gaussian noise along the plane antidiagonal (line
            with -pi/4 orientation):

                mean_F = np.array([0., 0.])
                cov_F = .2 * np.eye(2)
                mean_G = np.array([0., 0.])
                cov_G = np.array([
                    [1., -1.],
                    [-1., 1.],
                ])
                noisy_problem = nmoo.GaussianNoise(
                    problem,
                    {
                        "F": (mean_F, cov_F),
                        "G": (mean_G, cov_G),
                    },
                )

        See also:
            `pymoo documentation
            <https://pymoo.org/getting_started.html#By-Class>`_
        """
        super().__init__(problem, name=name)
        self._parameters = parameters
        self._generator = np.random.default_rng()

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
        noises: Dict[str, np.ndarray] = {}
        for k in self._parameters.keys():
            try:
                mean, cov = self._parameters[k]
                noises[k] = self._generator.multivariate_normal(
                    mean,
                    cov,
                    out[k].shape[0],
                ).reshape(out[k].shape)
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
