"""
A benchmarking utility
"""
__docformat__ = "google"

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import os

from joblib import delayed, Parallel
from pymoo.factory import get_performance_indicator
from pymoo.optimize import minimize
import numpy as np
import pandas as pd

from nmoo.callbacks import TimerCallback
from nmoo.indicators.delta_f import DeltaF
from nmoo.wrapped_problem import WrappedProblem


@dataclass
class Pair:
    """
    Represents a problem-algorithm pair and its result.
    """

    algorithm_description: Dict[str, Any]
    algorithm_name: str
    n_run: int
    problem_description: Dict[str, Any]
    problem_name: str
    result: Optional[pd.DataFrame] = None

    def __str__(self) -> str:
        return f"{self.problem_name} - {self.algorithm_name} ({self.n_run})"


# pylint: disable=too-many-instance-attributes
class Benchmark:
    """
    A benchmark is constructed with a list of problems and pymoo algorithms
    descriptions, and run each algorithm against each problem, storing all
    histories for later analysis.
    """

    SUPPORTED_PERFOMANCE_INDICATORS = [
        "df",
        "gd",
        "gd+",
        "hv",
        "igd",
        "igd+",
        "ps",
    ]

    _algorithms: Dict[str, dict]
    """
    List of algorithms to be benchmarked.
    """

    _dump_histories: bool
    """
    Wether the history of each `WrappedProblem` involved in this benchmark
    should be written to disk.
    """

    _max_retry: int
    """
    Maximum number of attempts to run a given problem-algorithm pair before
    giving up.
    """

    _n_runs: int
    """
    Number of times to run a given problem/algorithm pair.
    """

    _output_dir_path: Path
    """
    Path of the output directory.
    """

    _performance_indicators: List[str]
    """
    List of performance indicator to calculate during the benchmark.
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
        performance_indicators: Optional[List[str]] = None,
        max_retry: int = -1,
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
        * `df_n_evals` (int, optional): see the explanation of the `df`
            performance indicator below; defaults to `1`;
        * `evaluator` (optional): an algorithm evaluator object that will be
            applied to every algorithm that run on this problem; if an
            algorithm already has an evaluator attached to it (see
            `<algorithm_description>` below), the evaluator attached to this
            problem takes precedence; note that the evaluator is deepcopied for
            every run of `minimize`;
        * `hv_ref_point` (optional, `np.ndarray`): a reference point for
            computing hypervolume performance, see `performance_indicators`
            argument;
        * `pareto_front` (optional, `np.ndarray`): a Pareto front subset;
        * `problem`: a `WrappedProblem` instance.

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

        Args:
            algorithms (Dict[str, dict]): Dict of all algorithms to be
                benchmarked.
            dump_histories (bool): Wether the history of each
                `WrappedProblem` involved in this benchmark should be written
                to disk. Defaults to `True`.
            max_retries (int): Maximum number of attempts to run a given
                problem-algorithm pair before giving up. Set it to `-1` to
                retry indefinitely.
            n_runs (int): Number of times to run a given problem/algorithm
                pair.
            problems (Dict[str, dict]): Dict of all problems to be benchmarked.
            performance_indicators (Optional[List[str]]): List of perfomance
                indicators to be calculated and included in the result
                dataframe (see `Benchmark.final_results`). Supported indicators
                are
                * `df`: ΔF metric, see the documentation of
                    `nmoo.indicators.delta_f.DeltaF`;
                * `gd`: [generational distance](https://pymoo.org/misc/indicators.html#Generational-Distance-(GD)),
                    requires `pareto_front` to be set in the problem
                    description dictionaries, otherwise the value of this
                    indicator will be `NaN`;
                * `gd+`: [generational distance plus](https://pymoo.org/misc/indicators.html#Generational-Distance-Plus-(GD+)),
                    requires `pareto_front` to be set in the problem
                    description dictionaries, otherwise the value of this
                    indicator will be `NaN`;
                * `hv`: [hypervolume](https://pymoo.org/misc/indicators.html#Hypervolume),
                    requires `hv_ref_point` to be set in the problem
                    discription dictionaries, otherwise the value of this
                    indicator will be `NaN`;
                * `igd`: [inverted generational distance](https://pymoo.org/misc/indicators.html#Inverted-Generational-Distance-(IGD)),
                    requires `pareto_front` to be set in the problem
                    description dictionaries, otherwise the value of this
                    indicator will be `NaN`;
                * `igd+`: [inverted generational distance](https://pymoo.org/misc/indicators.html#Inverted-Generational-Distance-Plus-(IGD+)),
                    requires `pareto_front` to be set in the problem
                    description dictionaries, otherwise the value of this
                    indicator will be `NaN`;
                * `ps`: population size, or equivalently, the size of the
                    current Pareto front.

                In the result dataframe, the corresponding columns will be
                named `perf_<name of indicator>`, e.g. `perf_igd`. If left
                unspecified, defaults to `["gd", "gd+", "igd", "igd+"]`.
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
        self._max_retry = max_retry

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

        if performance_indicators is None:
            self._performance_indicators = ["gd", "gd+", "igd", "igd+"]
        else:
            self._performance_indicators = []
            for pi in set(performance_indicators):
                if pi not in Benchmark.SUPPORTED_PERFOMANCE_INDICATORS:
                    raise ValueError(f"Unknown performance indicator '{pi}'")
                self._performance_indicators.append(pi)
            self._performance_indicators = sorted(self._performance_indicators)

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

        columns = ["algorithm", "problem", "n_run", "n_gen", "timedelta"]
        columns += ["perf_" + pi for pi in self._performance_indicators]
        self._results = pd.DataFrame(columns=columns)

    def _run_pair(
        self,
        pair: Pair,
    ) -> Pair:
        """
        Runs a given algorithm against a given problem. See
        `nmoo.benchmark.Benchmark.run`. Immediately dumps the history of the
        problem and all wrapped problems with the following naming scheme:

            output_dir_path/<problem_name>.<algorithm_name>.<n_run>.<level>.npz

        where `level` is the depth of the wrapped problem, starting at `1`. See
        `nmoo.wrapped_problem.WrappedProblem.dump_all_histories`.

        Args:
            pair: A `Pair` object representint the problem-algorithm pair to
                run.

        Returns:
            The same `Pair` object, but with a populated `result` field if the
            minimzation procedure was successful.
        """
        logging.info("Running pair [%s]", pair)
        save_history = pair.algorithm_description.get("save_history", True)
        evaluator = pair.problem_description.get(
            "evaluator",
            pair.algorithm_description.get("evaluator"),
        )
        pair.problem_description["problem"].start_new_run()

        try:
            results = minimize(
                deepcopy(pair.problem_description["problem"]),
                pair.algorithm_description["algorithm"],
                termination=pair.algorithm_description.get("termination"),
                copy_algorithm=True,
                copy_termination=True,
                # extra Algorithm.setup kwargs
                callback=TimerCallback(),
                display=pair.algorithm_description.get("display"),
                evaluator=deepcopy(evaluator),
                return_least_infeasible=pair.algorithm_description.get(
                    "return_least_infeasible", False
                ),
                save_history=save_history,
                seed=pair.algorithm_description.get("seed"),
                verbose=pair.algorithm_description.get("verbose", False),
            )
        except:  # pylint: disable=bare-except
            logging.error("Pair [%s] failed. Rescheduling...", pair)
            return pair

        if self._dump_histories:
            results.problem.dump_all_histories(
                self._output_dir_path,
                f"{pair.problem_name}.{pair.algorithm_name}.{pair.n_run}",
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

        # pylint: disable=cell-var-from-loop
        for pi in self._performance_indicators:
            f = lambda _: np.nan
            if pi == "df":
                problem = pair.problem_description["problem"]
                if isinstance(problem, WrappedProblem):
                    n_evals = pair.problem_description.get("df_n_evals", 1)
                    delta_f = DeltaF(problem, n_evals)
                    f = lambda state: delta_f.do(
                        state.pop.get("F"), state.pop.get("X")
                    )
                else:
                    # the problem is already the ground problem
                    f = lambda _: 0.0
            elif pi in ["gd", "gd+", "igd", "igd+"]:
                # Pareto front defaults to last state optimal population's F
                pf = pair.problem_description.get(
                    "pareto_front", results.history[-1].opt.get("F")
                )
                ind = get_performance_indicator(pi, pf)
                f = lambda state: ind.do(state.pop.get("F"))
            elif pi == "hv" and "hv_ref_point" in pair.problem_description:
                hv = get_performance_indicator(
                    "hv", ref_point=pair.problem_description["hv_ref_point"]
                )
                f = lambda state: hv.do(state.pop.get("F"))
            elif pi == "ps":
                f = lambda state: len(state.opt.get("F"))
            df["perf_" + pi] = [f(state) for state in results.history]

        df["algorithm"] = pair.algorithm_name
        df["problem"] = pair.problem_name
        df["n_run"] = pair.n_run

        pair.result = df
        return pair

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
        pairs = [
            Pair(
                algorithm_description=aa,
                algorithm_name=an,
                n_run=r,
                problem_description=pp,
                problem_name=pn,
            )
            for (an, aa), (pn, pp), r in everything
        ]
        executor = Parallel(n_jobs=n_jobs, **joblib_kwargs)
        current_round = 0
        while (
            self._max_retry < 0 or current_round <= self._max_retry
        ) and len(pairs) > 0:
            results = executor(
                delayed(Benchmark._run_pair)(self, pair) for pair in pairs
            )
            pairs = []
            for pair in results:
                if pair.result is not None:
                    self._results = self._results.append(
                        pair.result,
                        ignore_index=True,
                    )
                else:
                    pairs.append(pair)
            current_round += 1
        if pairs:
            logging.warning(
                "Benchmark finished, but some pairs could not be run "
                "successfully within the retry budget (%d):",
                self._max_retry,
            )
            for pair in pairs:
                logging.warning("    %s", pair)
        self._results = self._results.astype(
            {
                "algorithm": "category",
                "n_gen": "uint32",
                "n_run": "uint32",
                "problem": "category",
                "timedelta": "timedelta64[ns]",
                **{
                    "perf_" + pi: "float64"
                    for pi in self._performance_indicators
                },
            }
        )
        self.dump_results(self._output_dir_path / "benchmark.csv", index=False)
