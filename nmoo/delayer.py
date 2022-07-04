"""
A problem that incurs a constant delay at each `_evaluate` call.
"""
__docformat__ = "google"

from time import sleep

from pymoo.core.problem import Problem
import numpy as np

from .wrapped_problem import WrappedProblem


class Delayer(WrappedProblem):
    """
    A problem that sleeps for a set amount of time at each `_evaluate` call,
    passing calling the wrapped problem's `_evaluate`.
    """

    _delay: float
    """
    Sleep time in seconds. Floating point number may be used to indicate a more
    precise sleep time.
    """

    def __init__(
        self,
        problem: Problem,
        delay: float = 0.05,
        *,
        name: str = "delayer",
        **kwargs,
    ):
        """
        Args:
            name (str): An optional name for this problem. This will be used
                when creating history dump files. Defaults to
                `wrapped_problem`.
        """
        super().__init__(problem, name=name, **kwargs)

        if delay < 0.0:
            raise ValueError("Delay must be a positive.")
        self._delay = delay

    def _evaluate(self, x, out, *args, **kwargs):
        sleep(self._delay)
        self._problem._evaluate(x, out, *args, **kwargs)
        self.add_to_history_x_out(
            x,
            out,
            delay=np.full((x.shape[0],), self._delay),
        )
