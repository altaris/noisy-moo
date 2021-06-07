"""
A benchmarking utility
"""
__docformat__ = "google"

from itertools import product
from pathlib import Path
from typing import Dict, Optional, Union

from pymoo.factory import get_performance_indicator
from pymoo.optimize import minimize
import pandas as pd

from nmoo.utils import TimerCallback


class Benchmark:
    """
    A benchmark is constructed with a list of problems and pymoo algorithms
    descriptions, and run each algorithm against each problem, storing all
    histories for later analysis.
    """

    _algorithms: Dict[str, dict]
    """
    List of algorithms to be benchmarked.
    """

    _n_runs: int
    """
    Number of times to run a given problem/algorithm pair.
    """

    _problems: Dict[str, dict]
    """
    List of problems to be benchmarked.
    """

    _results: pd.DataFrame
    """
    Results of all runs.
    """

    def __init__(
        self,
        problems: Dict[str, dict],
        algorithms: Dict[str, dict],
        n_runs: int = 1,
    ):
        """
        Constructor. The set of problems to be benchmarked is represented by a
        dictionary with the following structure::

            problems = {
                <problem_name>: <problem_description>,
                <problem_name>: <problem_description>,
            }

        where `<problem_name>` is a user-defined string (but stay reasonable since
        it may be used in filenames), and `<problem_description>` is a
        dictionary with the following keys:
        * `pareto_front` (optional, `np.ndarray`): a Pareto front subset.
        * `problem`: a `WrappedProblem` instance;

        The set of algorithms to be used is specified similarly::

            algorithms = {
                <algorithm_name>: <algorithm_description>,
                <algorithm_name>: <algorithm_description>,
            }

        where `<algorithm_name>` is a user-defined string (but stay reasonable
        since it may be used in filenames), and `<algorithm_description>` is
        a dictionary with the following keys:
        * `algorithm`: a pymoo `Algorithm` instance;
        * `seed` (optional, int): a seed.
        * `termination` (optional): a pymoo termination criterion;

        Args:
            algorithms (Dict[str, dict]): Dict of all algorithms to be
                benchmarked.
            n_runs (int): Number of times to run a given problem/algorithm
                pair.
            problems (Dict[str, dict]): Dict of all problems to be benchmarked.
        """
        if not algorithms:
            raise ValueError("A benchmark requires at least 1 algorithm.")
        for k, v in algorithms.items():
            if not isinstance(v, dict):
                raise ValueError(
                    f"Description for algorithm '{k}' must be a dict."
                )
            if "algorithm" not in v:
                raise ValueError(
                    f"Description for algorithm '{k}' is missing mandatory "
                    "key 'algorithm'."
                )
        self._algorithms = algorithms

        if n_runs <= 0:
            raise ValueError(
                "The number of run (for each problem/algorithm pair) must be "
                "at least 1."
            )
        self._n_runs = n_runs

        if not problems:
            raise ValueError("A benchmark requires at least 1 problem.")
        for k, v in problems.items():
            if not isinstance(v, dict):
                raise ValueError(
                    f"Description for problem '{k}' must be a dict."
                )
            if "problem" not in v:
                raise ValueError(
                    f"Description for problem '{k}' is missing mandatory key "
                    "'problem'."
                )
        self._problems = problems

        self._results = pd.DataFrame(
            columns=[
                "algorithm",
                "problem",
                "n_run",
                "n_gen",
                "timedelta",
                "perf_gd",
                "perf_gd+",
                "perf_igd",
                "perf_igd+",
            ],
        )

    def dump_everything(
        self,
        dir_path: Union[Path, str],
        benchmark_results_filename: str = "benchmark.csv",
        benchmark_results_fmt: str = "csv",
        benchmark_results_writer_kwargs: Optional[dict] = None,
        problem_histories_compressed: bool = True,
    ):
        """
        Dumps EVERYTHIIIING, i.e. the benchmark results (see
        `nmoo.benchmark.Benchmark.dump_results`) and all involved problems
        histories (see `nmoo.utils.WrappedProblem.dump_all_histories`).

        Args:
            dir_path (Union[Path, str]): Output directory
            benchmark_results_filename (str): Filename for the benchmark
                results. `benchmark.csv` by default.
            benchmark_results_fmt (str): Format of the benchmark results file,
                see `nmoo.benchmark.Benchmark.dump_results`. Defaults to CSV.
            benchmark_results_writer_kwargs (Optional[dict]): Optional kwargs
                to pass on to the `pandas.DataFrame.to_<fmt>` benchmark results
                writer method.
            problem_histories_compressed (bool): Wether to compress the problem
                history files, see `nmoo.utils.WrappedProblem.dump_history`.
        """
        if benchmark_results_writer_kwargs is None:
            benchmark_results_writer_kwargs = dict()
        self.dump_results(
            Path(dir_path) / benchmark_results_filename,
            benchmark_results_fmt,
            **benchmark_results_writer_kwargs,
        )
        for pn, pp in self._problems.items():
            pp["problem"].dump_all_histories(
                dir_path,
                pn,
                problem_histories_compressed,
            )

    def dump_results(self, path: Union[Path, str], fmt: str = "csv", **kwargs):
        """
        Dumps the internal `_result` dataframe.

        Args:
            path (Union[Path, str]): Path to the output file.
            fmt (str): Text or binary format supported by pandas, see
                `here <https://pandas.pydata.org/docs/user_guide/io.html>`_.
                CSV by default.
            kwargs: Will be passed on the `pandas.DataFrame.to_<fmt>` method.
        """
        saver = {
            "csv": pd.DataFrame.to_csv,
            "excel": pd.DataFrame.to_excel,
            "feather": pd.DataFrame.to_feather,
            "gbq": pd.DataFrame.to_gbq,
            "hdf": pd.DataFrame.to_hdf,
            "html": pd.DataFrame.to_html,
            "json": pd.DataFrame.to_json,
            "parquet": pd.DataFrame.to_parquet,
            "pickle": pd.DataFrame.to_pickle,
        }[fmt]
        saver(self._results, path, **kwargs)

    def final_results(
        self,
        timedeltas_to_microseconds: bool = True,
        reset_index: bool = True,
    ) -> pd.DataFrame:
        """
        Returns a dataframe containing the final row of each
        algorithm/problem/n_run triple, i.e. the final record of each run of
        the benchmark.

        If the `reset_index` argument set to `False`, the resulting dataframe
        will have a multiindex given by the (algorithm, problem, n_run) tuples,
        e.g.

                                     n_gen  timedelta   perf_gd  ...
            algorithm problem n_run                              ...
            nsga2     bnh     1        155     886181  0.477980  ...
                              2        200      29909  0.480764  ...
                      zdt1    1        400     752818  0.191490  ...
                              2        305     979112  0.260930  ...

        (note tha the `timedelta` column has been converted to microseconds,
        see the `timedeltas_to_microseconds` argument below). If `reset_index`
        is set to `True` (the default), then the index is reset, giving
        something like this:

              algorithm problem  n_run  n_gen  timedelta   perf_gd  ...
            0     nsga2     bnh      1    155     886181  0.477980  ...
            1     nsga2     bnh      2    200      29909  0.480764  ...
            2     nsga2    zdt1      1    400     752818  0.191490  ...
            3     nsga2    zdt1      2    305     979112  0.260930  ...

        This form is easier to plot.

        Args:
            reset_index (bool): Wether to reset the index. Defaults to
                `True`.
            timedeltas_to_microseconds (bool): Wether to convert the
                timedeltas column to microseconds. Defaults to `True`.

        """
        df = self._results.groupby(["algorithm", "problem", "n_run"]).last()
        if timedeltas_to_microseconds:
            df["timedelta"] = df["timedelta"].dt.microseconds
        return df.reset_index() if reset_index else df

    def run(self):
        """
        Runs the benchmark sequentially. Makes your laptop go brr.
        """
        everything = product(
            self._algorithms.items(),
            self._problems.items(),
            range(1, self._n_runs + 1),
        )
        n_pairs = len(self._problems) * len(self._algorithms) * self._n_runs
        for i, ((an, aa), (pn, pp), r) in enumerate(everything):
            print(
                f"[{i+1}/{n_pairs}] Problem: {pn}, Algorithm: {an}, "
                f"Run: {r}/{self._n_runs}"
            )
            pp["problem"].start_new_run()
            results = minimize(
                pp["problem"],
                aa["algorithm"],
                aa.get("termination", None),
                callback=TimerCallback(),
                save_history=True,
                seed=aa.get("seed", None),
                verbose=False,
            )
            df = pd.DataFrame()
            df["n_gen"] = range(1, len(results.history) + 1)
            df["timedelta"] = results.algorithm.callback._deltas
            if "pareto_front" in pp:
                for pi in ["gd", "gd+", "igd", "igd+"]:
                    ind = get_performance_indicator(pi, pp["pareto_front"])
                    df["perf_" + pi] = [
                        # TODO: Generalize
                        ind.calc(state.pop.get("F"))
                        for state in results.history
                    ]
            df["algorithm"] = an
            df["problem"] = pn
            df["n_run"] = r
            self._results = self._results.append(df, ignore_index=True)
