"""
Various utilities.
"""
__docformat__ = "google"

from typing import Dict

from pymoo.model.problem import Problem
import numpy as np


class ProblemWrapper(Problem):
    """
    A simple Pymoo :obj:`Problem` wrapper that keeps a history of all calls made
    to it.
    """

    _history: Dict[str, np.ndarray]
    """
    A history is a dictionary that maps string keys (e.g. `"x"`) to a numpy
    array of all values of that key (e.g. all values of `"x"`). All the numpy
    arrays should have the same length (0th shape component) but are not
    required to have the same type.

    If you subclass this class, don't forget to carefully document the meaning
    of the keys of what you're story in history.
    """

    _problem: Problem
    """Wrapped pymoo problem."""

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
        self._history = dict()
        self._problem = problem

    def add_to_history(self, **kwargs):
        """
        Adds records to the history. The provided keys should match that of the
        history (this is not checked at runtime for perfomance reasons). The
        provided values should be numpy arrays that all have the same length
        (0th shape component).
        """
        for k, v in kwargs.items():
            if k not in self._history:
                self._history[k] = v.copy()
            else:
                self._history[k] = np.append(
                    self._history[k], v.copy(), axis=0
                )

    def add_to_history_x_out(self, x: np.ndarray, out: dict, **kwargs):
        """
        Convenience function to add the `_evaluate` method's `x` and `out` to
        history, along with potentially other items.
        """
        self.add_to_history(
            x=x,
            **{k: v for k, v in out.items() if isinstance(v, np.ndarray)},
            **kwargs,
        )

    def dump_history(self, path: str, compressed: bool = True):
        """
        Dumps the history into an NPZ archive.

        Args:
            path (str): File path of the output archive.
            compressed (bool): Wether to compress the archive (defaults to
                `True`).

        See also:
            https://numpy.org/doc/stable/reference/generated/numpy.load.html
        """
        saver = np.savez_compressed if compressed else np.savez
        saver(path, **self._history)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Calls the wrapped problems's `_evaluate` method and appends its input
        (`x`) and output (`out`) to the history.
        """
        self._problem._evaluate(x, out, *args, **kwargs)
        self.add_to_history_x_out(x, out)
