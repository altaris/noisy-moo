"""
Base `nmoo` problem class. A `WrappedProblem` is simply a
[`pymoo.core.problem.Problem`](https://pymoo.org/problems/definition.html) that
contains another problem (`nmoo` or `pymoo`) to which calls to `_evaluate` are
deferred to. A `WrappedProblem` also have call history features (although it is
the responsability of the `_evaluate` implementation to populate it).

Note:
    Since `WrappedProblem` directly inherits from
    [`pymoo.core.problem.Problem`](https://pymoo.org/problems/definition.html),
    wrapped problems can be used seemlessly with `pymoo`.
"""
__docformat__ = "google"

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
from pymoo.core.problem import Problem


class WrappedProblem(Problem):
    """
    A simple Pymoo :obj:`Problem` wrapper that keeps a history of all calls
    made to it.
    """

    _current_history_batch: int = 0
    """
    The number of batches added to history.
    """

    _current_run: int = 0
    """
    Current run number. See `nmoo.utils.WrappedProblem.start_new_run`.
    """

    _history: Dict[str, np.ndarray]
    """
    A history is a dictionary that maps string keys (e.g. `"X"`) to a numpy
    array of all values of that key (e.g. all values of `"X"`). All the numpy
    arrays should have the same length (0th shape component) but are not
    required to have the same type.

    If you subclass this class, don't forget to carefully document the meaning
    of the keys of what you're story in history.
    """

    _name: str
    """The name of this problem."""

    _problem: Problem
    """Wrapped pymoo problem."""

    def __init__(
        self,
        problem: Problem,
        *,
        copy_problem: bool = True,
        name: str = "wrapped_problem",
    ):
        """
        Constructor.

        Args:
            copy_problem (bool): Wether to deepcopy the problem. If `problem`
                is a `WrappedProblem`, this is recommended to avoid history
                clashes. However, if whatever benchmark that uses this problem
                uses multiprocessing (as opposed to single or multithreading),
                this does not seem to be necessary.
            problem (:obj:`Problem`): A non-noisy pymoo problem.
            name (str): An optional name for this problem. This will be used
                when creating history dump files. Defaults to
                `wrapped_problem`.
        """
        super().__init__(
            n_var=problem.n_var,
            n_obj=problem.n_obj,
            n_constr=problem.n_constr,
            xl=problem.xl,
            xu=problem.xu,
            check_inconsistencies=problem.check_inconsistencies,
            replace_nan_values_by=problem.replace_nan_values_by,
            exclude_from_serialization=problem.exclude_from_serialization,
            callback=problem.callback,
        )
        self._history = {}
        self._name = name
        self._problem = deepcopy(problem) if copy_problem else problem

    def __str__(self):
        return self._name

    def add_to_history(self, **kwargs):
        """
        Adds records to the history. The provided keys should match that of the
        history (this is not checked at runtime for perfomance reasons). The
        provided values should be numpy arrays that all have the same length
        (0th shape component).
        """
        self._current_history_batch += 1
        if not kwargs:
            # No items to add
            return
        lengths = {k: v.shape[0] for k, v in kwargs.items()}
        if len(set(lengths.values())) > 1:
            logging.warning(
                "[add_to_history] The lengths of the arrays don't match: %s",
                str(lengths),
            )
        kwargs["_batch"] = np.full(
            (max(lengths.values()),),
            self._current_history_batch,
        )
        kwargs["_run"] = np.full(
            (max(lengths.values()),),
            self._current_run,
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
        history, along with potentially other items. Note that the `x` argument
        is stored under the `X` key to remain consistent with `pymoo`'s API.
        """
        self.add_to_history(
            X=x,
            **{k: v for k, v in out.items() if isinstance(v, np.ndarray)},
            **kwargs,
        )

    def all_layers(self) -> List["WrappedProblem"]:
        """
        Returns a list of all the `nmoo.wrapped_problem.WrappedProblem` wrapped
        within (including the current one). This list is ordered from the
        outermost one (the current problem) to the innermost one.
        """
        return [self] + (
            self._problem.all_layers()
            if isinstance(self._problem, WrappedProblem)
            else []
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
        instances of `nmoo.utils.WrappedProblem`) recursively wrapped within
        it. This will result in one `.npz` for each problem involved.

        Args:
            dir_path (Union[Path, str]): Output directory.
            name (str): The output files will be named according to the
                following pattern: `<name>.<_idx>-<layer_name>.npz`, where
                `_idx` is the "depth" of the corresponding problem (1 for the
                outermost, 2 for the one wrapped within it, etc.), and
                `layer_name` is the name of the current `WrappedProblem`
                instance (see `WrappedProblem.__init__`).
            compressed (bool): Wether to compress the archive (defaults to
                `True`).
            _idx (int): Don't touch that.

        See also:
            `nmoo.utils.WrappedProblem.dump_history`
        """
        self.dump_history(
            Path(dir_path) / f"{name}.{_idx}-{self._name}.npz", compressed
        )
        if isinstance(self._problem, WrappedProblem):
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

    def ground_problem(self) -> Problem:
        """
        Recursively goes down the problem wrappers until an actual
        `pymoo.Problem` is found, and returns it.
        """
        if isinstance(self._problem, WrappedProblem):
            return self._problem.ground_problem()
        return self._problem

    def reseed(self, seed: Any) -> None:
        """
        Recursively resets the internal random state of the problem. See the
        [numpy
        documentation](https://numpy.org/doc/stable/reference/random/generator.html?highlight=default_rng#numpy.random.default_rng)
        for details about acceptable seeds.
        """
        if isinstance(self._problem, WrappedProblem):
            self._problem.reseed(seed)

    def start_new_run(self):
        """
        In short, it rotates the history of the current problem, and all
        problems wrapped within.

        Every entry in the history is annotated with a run number. If this
        problem is reused (e.g. during benchmarks), you can call this method to
        increase the run number for all subsequent history entries.
        Additionally, the `_current_history_batch` counter is reset to `0`.

        If the wrapped problem is itself a `WrappedProblem`, then this method
        is recursively called.
        """
        self._current_run += 1
        self._current_history_batch = 0
        if isinstance(self._problem, WrappedProblem):
            self._problem.start_new_run()

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Calls the wrapped problems's `_evaluate` method and appends its input
        (`x`) and output (`out`) to the history.
        """
        self._problem._evaluate(x, out, *args, **kwargs)
        self.add_to_history_x_out(x, out)
