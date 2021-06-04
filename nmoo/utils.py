"""
Various utilities.
"""
__docformat__ = "google"

from pathlib import Path
from typing import Dict, List, Union
import logging

from pymoo.model.callback import Callback
from pymoo.model.problem import Problem
import numpy as np
import pandas as pd


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

    _history_batch_count: int = 0
    """
    The number of batches added to history.
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
        self._history_batch_count += 1
        if not kwargs:
            # No items to add
            return
        lengths = {k: v.shape[0] for k, v in kwargs.items()}
        if len(set(lengths.values())) > 1:
            logging.warn(
                "[add_to_history] The lengths of the arrays don't match: "
                + str(lengths)
            )
        kwargs["_batch"] = np.full(
            (max(lengths.values()),),
            self._history_batch_count,
        )
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

    def dump_all_histories(
        self,
        dir_path: Union[Path, str],
        name: str,
        compressed: bool = True,
        _idx: int = 1,
    ):
        """
        Dumps this problem's history, as well as all problems (more precisely,
        instances of `nmoo.utils.ProblemWrapper`) recursively wrapped within
        it. This will result in one `.npz` for each problem involved.

        Args:
            dir_path (Union[Path, str]): Output directory.
            name (str): The output files will be named according to the
                following pattern: `<name>.<_idx>.npz`, where `_idx` is the
                "depth" of the corresponding problem (1 for the outermost, 2
                for the one wrapped within it, etc.). Note that the `.npz`
                extension is automatically added.
            compressed (bool): Wether to compress the archive (defaults to
                `True`).
            _idx (int): Don't touch that.

        See also:
            `nmoo.utils.ProblemWrapper.dump_history`
        """
        self.dump_history(Path(dir_path) / f"{name}.{_idx}.npz", compressed)
        if isinstance(self._problem, ProblemWrapper):
            self._problem.dump_all_histories(
                dir_path, name, compressed, _idx + 1
            )

    def dump_history(self, path: Union[Path, str], compressed: bool = True):
        """
        Dumps the history into an NPZ archive.

        Args:
            path (Union[Path, str]): File path of the output archive.
            compressed (bool): Wether to compress the archive (defaults to
                `True`).

        See also:
            `numpy.load
            <https://numpy.org/doc/stable/reference/generated/numpy.load.html>`_

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


class TimerCallback(Callback):
    """
    A simple callback that register a timedelta (from the time it was
    instanciated) everytime it is notified.
    """

    _deltas: List[pd.Timedelta]
    """
    Timedeltas.
    """

    _initial_time: pd.Timestamp
    """
    Timestamp of when this instance has been created.
    """

    def __init__(self):
        self._deltas = []
        self._initial_time = pd.Timestamp.now()

    def notify(self, algorithm, **kwargs):
        self._deltas.append(pd.Timestamp.now() - self._initial_time)
