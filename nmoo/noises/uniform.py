"""
Wrapper that generates a uniform noise.
"""
__docformat__ = "google"

import logging
from typing import Any, Dict, List, Tuple, Union

from pymoo.core.problem import Problem
import numpy as np

from nmoo.wrapped_problem import WrappedProblem


class UniformNoise(WrappedProblem):
    """
    A wrapper that adds a uniform noise to a problem.

    Assume that the output of the wrapped problem as an `F` numerical component
    (as they almost always do) of dimension 2. The following creates a new
    problem by adding a `U(-1, 1)` noise on all components of `F` (without any
    covariance):

        noisy_problem = nmoo.UniformNoise(problem, 1)

    The following adds a `U(-1, 1)` noise on the first component of `F` but a
    `U(-2, 2)` noise on the second:

        noisy_problem = nmoo.UniformNoise(problem, [1, 2])

    For biased noises (i.e. with nonzero mean), the min and max values of every
    distribution must be specified. For example the following adds a `U(-.1,
    1)` noise on the first component of `F` but a `U(-.2, 2)` noise on the
    second:

        noisy_problem = nmoo.UniformNoise(problem, [[-.1, 1], [-.2, 2]])

    Note that mixed bound specifications such as

        noisy_problem = nmoo.UniformNoise(problem, [[-.1, 1], 2])
        # instead of [[-.1, 1], [-2, 2]]

    is not possible.

    If you want to add noise to more outputs of the wrapped problem, a
    parameter specification like `nmoo.noises.GaussianNoise` is also possible.
    For example, assume that the problem has a `G` numerical component. To
    apply a `U(-3, 3)` to all components of `G`, together with the noise above
    for `F`,

        noisy_problem = nmoo.UniformNoise(
            problem,
            {
                "F": [[-.1, 1], [-.2, 2]],
                "G": 3,
            }
        )

    """

    _generator: np.random.Generator
    """Random number generator."""

    _parameters: Dict[str, List[Tuple[float, float]]] = {}
    """Noise parameters."""

    def __init__(
        self,
        problem: Problem,
        parameters: Union[
            float,
            Tuple[float, float],
            List[Tuple[float, float]],
            Dict[
                str,
                Union[
                    float,
                    Tuple[float, float],
                    List[Tuple[float, float]],
                ],
            ],
        ],
        seed: Any = None,
        *,
        name: str = "uniform_noise",
        **kwargs,
    ):
        """
        Args:
            name (str): An optional name for this problem. This will be used
                when creating history dump files. Defaults to `gaussian_noise`.
            problem (:obj:`Problem`): A non-noisy pymoo problem.
            parameters: See the examples above.
            seed: Seed for
                [`numpy.random.default_rng`](https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.default_rng)
        """
        super().__init__(problem, name=name, **kwargs)
        if not isinstance(parameters, dict):
            parameters = {"F": parameters}
        try:
            for k, v in parameters.items():
                if isinstance(v, (float, int)):
                    assert v >= 0
                    self._parameters[k] = [(-v, v)]
                elif isinstance(v, (list, tuple)) and isinstance(v[0], list):
                    # v expected to be a list of min-max tuples
                    for w in v:
                        assert isinstance(w, (list, tuple))
                        assert len(w) == 2
                        assert w[0] <= w[1]
                    self._parameters[k] = parameters[k]
                else:
                    # v is expected to be a list of numbers
                    self._parameters[k] = []
                    for a in v:
                        assert isinstance(a, (float, int))
                        assert a >= 0
                        self._parameters[k].append((-a, a))
        except AssertionError as e:
            raise ValueError("Invalid noise parameters") from e
        self.reseed(seed)

    # pylint: disable=duplicate-code
    def _evaluate(self, x, out, *args, **kwargs):
        """
        Calls the wrapped problems's `_evaluate` method and adds a uniform
        noise. The history scheme is the same as in
        `nmoo.noises.GaussianNoise._evaluate`.
        """
        self._problem._evaluate(x, out, *args, **kwargs)
        noises: Dict[str, np.ndarray] = {}
        for k, v in self._parameters.items():
            try:
                noises[k] = np.array(
                    [self._generator.uniform(*b) for b in v]
                ).reshape(out[k].shape)
                out[k] += noises[k]
            except KeyError:
                logging.error(
                    "Noise parameter key %s is not present in objective "
                    "function output keys. No noise will be applied. "
                    "Objective function keys: %s.",
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
