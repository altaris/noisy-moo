"""
Various utilities.
"""

from pymoo.model.problem import Problem
import numpy as np
import pandas as pd


def np2d_to_df(array: np.ndarray, prefix: str) -> pd.DataFrame:
    """
    Converts a 2D numpy array to a Pandas :obj:`DataFrame`, where the column
    names are derived from the `prefix` by adding `_<col_index>`.

    Example:

        >>> np2d_to_df(np.ones((3, 4)), "x")

           x_0  x_1  x_2  x_3
        0  1.0  1.0  1.0  1.0
        1  1.0  1.0  1.0  1.0
        2  1.0  1.0  1.0  1.0
    """
    columns = [prefix + "_" + str(i) for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns)


def x_out_to_df(x: np.ndarray, out: dict) -> pd.DataFrame:
    """
    Converts the `x` and `out` from the :Problem._evaluate: callback to a
    single Pandas :obj:`DataFrame`.
    """
    df = np2d_to_df(x, "x")
    for k, v in out.items():
        if isinstance(v, np.ndarray):
            df = pd.concat([df, np2d_to_df(v, k)], axis=1)
        else:
            df[k] = v
    return df


class ProblemWrapper(Problem):
    """
    A noise class is a pymoo problem wrapping another (non noisy) problem.
    """

    _history = pd.DataFrame()
    """
    Dataframe containing the history of all `_evaluate` calls and potentially
    additional data.
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
        self._problem = problem

    def add_to_history(self, df: pd.DataFrame):
        """
        Adds records (in the form of a Pandas :obj:`DataFrame`) to the history.
        """
        # df["timestamp"] = np.datetime64("now")
        self._history = self._history.append(df, ignore_index=True)

    def _evaluate(self, x, out, *args, **kwargs):
        self._problem._evaluate(x, out, *args, **kwargs)
        self.add_to_history(x_out_to_df(x, out))
