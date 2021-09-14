"""
A benchmarking utility
"""
__docformat__ = "google"

from copy import deepcopy
import os

from itertools import product
from pathlib import Path
from typing import Dict, Union
import logging

from joblib import delayed, Parallel
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

    _dump_histories: bool
    """
    Wether the history of each `WrappedProblem` involved in this benchmark
    should be written to disk.
    """

    _n_runs: int
    """
    Number of times to run a given problem/algorithm pair.
    """

    _output_dir_path: Path
    """
    Path of the output directory.
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
        output_dir_path: Union[Path, str],
        problems: Dict[str, dict],
        algorithms: Dict[str, dict],
        n_runs: int = 1,
        dump_histories: bool = True,
    ):
        """
        Constructor. The set of problems to be benchmarked is represented by a
        dictionary with the following structure::

            problems = {
                <problem_name>: <problem_description>,
                <problem_name>: <problem_description>,
            }

        where `<problem_name>` is a user-defined string (but stay reasonable
        since it may be used in filenames), and `<problem_description>` is a
        dictionary with the following keys:
        * `evaluator` (optional): an algorithm evaluator object that will be
            applied to every algorithm that run on this problem; if an
            algorithm already has an evaluator attached to it (see
            `<algorithm_description>` below), the evaluator attached to this
            problem takes precedence; note that the evaluator is deepcopied for
            every run of `minimize`;
        * `pareto_front` (optional, `np.ndarray`): a Pareto front subset.
        * `problem`: a `WrappedProblem` instance;

        The set of algorithms to be used is specified similarly::

            algorithms = {
                <algorithm_name>: <algorithm_description>,
                <algorithm_name>: <algorithm_description>,
            }

        where `<algorithm_name>` is a user-defined string (but stay reasonable
        since it may be used in filenames), and `<algorithm_description>` is a
        dictionary with the following keys:
        * `algorithm`: a pymoo `Algorithm` object; note that it is deepcopied
            for every run of `minimize`;
        * `display` (optional): a custom `pymoo.util.display.Display` object
            for customization purposes;
        * `evaluator` (optional): an algorithm evaluator object; note that it
            is deepcopied for every run of `minimize`;
        * `return_least_infeasible` (optional, bool): if the algorithm cannot
            find a feasable solution, wether the least infeasable solution
            should still be returned; defaults to `False`;
        * `save_history` (optional, bool): wether a snapshot of the algorithm
            object should be kept at each iteration; defaults to `True`;
        * `seed` (optional, int): a seed;
        * `termination` (optional): a pymoo termination criterion; note that it
            is deepcopied for every run of `minimize`;
        * `verbose` (optional, bool): wether outputs should be printed during
            during the execution of the algorithm; defaults to `False`.

        Args: algorithms (Dict[str, dict]): Dict of all algorithms to be
            benchmarked. dump_histories (bool): Wether the history of each
            `WrappedProblem` involved in this benchmark should be written to
            disk. Defaults to `True`. n_runs (int): Number of times to run a
            given problem/algorithm pair. problems (Dict[str, dict]): Dict of
            all problems to be benchmarked.
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

        self._dump_histories = dump_histories

        if n_runs <= 0:
            raise ValueError(
                "The number of run (for each problem/algorithm pair) must be "
                "at least 1."
            )
        self._n_runs = n_runs

        if not os.path.isdir(output_dir_path):
            raise ValueError(
                f"Output directory '{output_dir_path}' does not exist"
            )
        self._output_dir_path = Path(output_dir_path)

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

    def _run_pair(
        self,
        algorithm_name: str,
        algorithm_desciption: dict,
        problem_name: str,
        problem_description: dict,
        n_run: int,
        n_run_global: int,
    ) -> pd.DataFrame:
        """
        Runs a given algorithm against a given problem. See
        `nmoo.benchmark.Benchmark.run`. Immediately dumps the history of the
        problem and all wrapped problems with the following naming scheme:

            output_dir_path/<problem_name>.<algorithm_name>.<n_run>.<level>.npz

        where `level` is the depth of the wrapped problem, starting at `1`. See
        `nmoo.wrapped_problem.WrappedProblem.dump_all_histories`.

        Args:
            algorithm_name (str): Algorithm name.
            algorithm_desciption (dict): Algorithm description
                dictionary, see `nmoo.benchmark.Benchmark.__init__`.
            problem_name (str): Problem name.
            problem_description (dict): Problem description
                dictionary, see `nmoo.benchmark.Benchmark.__init__`.
            n_run (int): Run number of that given algorithm/problem pair.
            n_run_global (int): Global run number (across all pairs
                algorithm/problem pairs).

        Returns:
            Run result as a pandas `DataFrame`.
        """
        n_pairs = len(self._problems) * len(self._algorithms) * self._n_runs
        print(
            f"[{n_run_global+1}/{n_pairs}] Problem: {problem_name}, "
            f"Algorithm: {algorithm_name}, Run: {n_run}/{self._n_runs}"
        )
        save_history = algorithm_desciption.get("save_history", True)
        evaluator = problem_description.get(
            "evaluator",
            algorithm_desciption.get("evaluator"),
        )
        problem_description["problem"].start_new_run()
        results = minimize(
            deepcopy(problem_description["problem"]),
            algorithm_desciption["algorithm"],
            termination=algorithm_desciption.get("termination"),
            copy_algorithm=True,
            copy_termination=True,
            # extra Algorithm.setup kwargs
            callback=TimerCallback(),
            display=algorithm_desciption.get("display"),
            evaluator=deepcopy(evaluator),
            return_least_infeasible=algorithm_desciption.get(
                "return_least_infeasible", False
            ),
            save_history=save_history,
            seed=algorithm_desciption.get("seed"),
            verbose=algorithm_desciption.get("verbose", False),
        )

        if self._dump_histories:
            results.problem.dump_all_histories(
                self._output_dir_path,
                f"{problem_name}.{algorithm_name}.{n_run}",
            )

        df = pd.DataFrame()

        if not save_history:
            results.history = [results.algorithm]

        df["n_gen"] = [a.n_gen for a in results.history]
        df["timedelta"] = (
            results.algorithm.callback._deltas
            if save_history
            else [results.algorithm.callback._deltas[-1]]
        )

        if "pareto_front" in problem_description:
            pareto_front = problem_description["pareto_front"]
            if pareto_front is None:
                logging.error(
                    "Specified pareto front for problem %s is None",
                    str(problem_description["problem"]),
                )
            else:
                for pi in ["gd", "gd+", "igd", "igd+"]:
                    ind = get_performance_indicator(
                        pi, problem_description["pareto_front"]
                    )
                    df["perf_" + pi] = [
                        # TODO: Generalize
                        ind.calc(state.pop.get("F"))
                        for state in results.history
                    ]

        df["algorithm"] = algorithm_name
        df["problem"] = problem_name
        df["n_run"] = n_run

        return df

    # def dump_everything(
    #     self,
    #     dir_path: Union[Path, str],
    #     benchmark_results_filename: str = "benchmark.csv",
    #     benchmark_results_fmt: str = "csv",
    #     benchmark_results_writer_kwargs: Optional[dict] = None,
    #     problem_histories_compressed: bool = True,
    # ):
    #     """
    #     Dumps EVERYTHIIIING, i.e. the benchmark results (see
    #     `nmoo.benchmark.Benchmark.dump_results`) and all involved problems
    #     histories (see `nmoo.utils.WrappedProblem.dump_all_histories`).

    #     Args:
    #         dir_path (Union[Path, str]): Output directory
    #         benchmark_results_filename (str): Filename for the benchmark
    #             results. `benchmark.csv` by default.
    #         benchmark_results_fmt (str): Format of the benchmark results file,
    #             see `nmoo.benchmark.Benchmark.dump_results`. Defaults to CSV.
    #         benchmark_results_writer_kwargs (Optional[dict]): Optional kwargs
    #             to pass on to the `pandas.DataFrame.to_<fmt>` benchmark results
    #             writer method.
    #         problem_histories_compressed (bool): Wether to compress the problem
    #             history files, see `nmoo.utils.WrappedProblem.dump_history`.
    #     """
    #     if benchmark_results_writer_kwargs is None:
    #         benchmark_results_writer_kwargs = dict()
    #     self.dump_results(
    #         Path(dir_path) / benchmark_results_filename,
    #         benchmark_results_fmt,
    #         **benchmark_results_writer_kwargs,
    #     )
    #     for pn, pp in self._problems.items():
    #         pp["problem"].dump_all_histories(
    #             dir_path,
    #             pn,
    #             problem_histories_compressed,
    #         )

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

    def run(self, n_jobs: int = -1, **joblib_kwargs):
        """
        Runs the benchmark sequentially. Makes your laptop go brr. The
        histories of all problems are progressively dumped in the specified
        output directory as the benchmark run. At the end, the benchmark
        results are dumped in `output_dir_path/benchmark.csv`.

        Args:
            n_jobs (int): Number of threads to use. See the `joblib.Parallel`_
                documentation. Defaults to `-1`, i.e. all CPUs are used.
            joblib_kwargs (dict): Additional kwargs to pass on to the
                `joblib.Parallel`_ instance.

        .. _joblib.Parallel:
            https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
        """
        everything = product(
            self._algorithms.items(),
            self._problems.items(),
            range(1, self._n_runs + 1),
        )
        executor = Parallel(n_jobs=n_jobs, **joblib_kwargs)
        results = executor(
            delayed(Benchmark._run_pair)(self, an, aa, pn, pp, r, i)
            for i, ((an, aa), (pn, pp), r) in enumerate(everything)
        )
        for df in results:
            self._results = self._results.append(df, ignore_index=True)
        self.dump_results(self._output_dir_path / "benchmark.csv")
