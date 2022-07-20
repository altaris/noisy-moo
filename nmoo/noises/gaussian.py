"""
Random noises to apply to objective functions.
"""
__docformat__ = "google"

import logging
from typing import Any, Dict, Optional, Tuple, Union

from pymoo.core.problem import Problem
import numpy as np

from nmoo.wrapped_problem import WrappedProblem


class GaussianNoise(WrappedProblem):
    """
    A wrapper that adds a (multivariate) gaussian noise to a problem.

    Assume that the output of the wrapped problem as an `F` numerical component
    (as they almost always do). The following creates a new problem by adding a
    `N(0, .2)` noise on all components of `F` (without any covariance):

        mean_F = np.array([0., 0.])
        cov_F = .2 * np.eye(2)
        noisy_problem = nmoo.GaussianNoise(problem, mean_F, cov_F)

    Assume that in addition, the problem has a `G` numerical component to which
    we would also like to add noise. The following a 0-mean 1-dimensional
    gaussian noise along the plane antidiagonal (line with -pi/4 orientation)
    to `G`, and the same noise as above to `F`:

        mean_F = np.array([0., 0.])
        cov_F = .2 * np.eye(2)
        mean_G = np.array([0., 0.])
        cov_G = np.array([[1., -1.], [-1., 1.]])
        noisy_problem = nmoo.GaussianNoise(
            problem, {
                "F": (mean_F, cov_F),
                "G": (mean_G, cov_G),
            },
        )

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
        mean: Optional[np.ndarray] = None,
        covariance: Optional[Union[float, int, np.ndarray]] = None,
        parameters: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        seed: Any = None,
        *,
        name: str = "gaussian_noise",
        **kwargs,
    ):
        """
        Args:
            name (str): An optional name for this problem. This will be used
                when creating history dump files. Defaults to `gaussian_noise`.
            problem (:obj:`Problem`): A non-noisy pymoo problem.
            mean (optional `np.ndarray`): The mean vector of the gaussian
                distribution. If specified, the `covariance` argument must also
                be specified, and `parameters` must be left to its default
                `None`.
            covariance (optional `np.ndarray` or number): The covariance
                matrix of the gaussian distribution. If specified, the `mean`
                argument must also be specified, and `parameters` must be left
                to its default `None`. For convenience, a number `v` can be
                passed instead of a matrix, in which case the covariance matrix
                is set to be `v * I_n`, where `n` is the dimension of the
                `mean` vector. Note that `v` is then the variance of every
                component of the distribution, **not the standard deviation**!
            parameters (optional dict): Gaussian noise parameters, in the form
                of a dict mapping the name of an objective to a numpy array
                pair (mean, covariance matrix). The set of keys should be a
                subset of the final `out` dictionary keys in the wrapped
                problem's `_evaluate` method. If specified, the `mean` and
                `covariance` arguments must be left to their default `None`.
            seed: Seed for
                [`numpy.random.default_rng`](https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.default_rng)
        """
        super().__init__(problem, name=name, **kwargs)
        if mean is not None and covariance is not None and parameters is None:
            if not isinstance(covariance, np.ndarray):
                covariance = covariance * np.eye(mean.shape[0])
            self._parameters = {"F": (mean, covariance)}
        elif mean is None and covariance is None and parameters is not None:
            self._parameters = parameters
        else:
            raise ValueError(
                "Invalid noise specification. Either mean and covariance are "
                "both set, or a parameters dict is set."
            )
        self.reseed(seed)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Calls the wrapped problems's `_evaluate` method and adds a Gaussian
        noise. Adds the input (`x`), the noisy output, and the noise values to
        history.

        Example:

            If the wrapped problem's output dict looks like:

                {
                    "A": (n, m) np.ndarray,
                    "B": not an np.ndarray
                }

            then the history will look like this:

                {
                    "A": an np.ndarray (with the noise added),
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

    def reseed(self, seed: Any) -> None:
        self._generator = np.random.default_rng(seed)
        if isinstance(self._problem, WrappedProblem):
            self._problem.reseed(seed)
