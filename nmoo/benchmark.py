"""
A benchmarking utility
"""
__docformat__ = "google"

import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pymoo.core.population import Population
from pymoo.factory import get_performance_indicator
from pymoo.optimize import minimize
from pymoo.util.optimum import filter_optimum

from nmoo.callbacks import TimerCallback
from nmoo.indicators.delta_f import DeltaF


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

    def __str__(self) -> str:
        return f"{self.problem_name} - {self.algorithm_name} ({self.n_run})"

    def filename_prefix(self) -> str:
        """Returns `<problem_name>.<algorithm_name>.<n_run>`."""
        return f"{self.problem_name}.{self.algorithm_name}.{self.n_run}"

    def pareto_population_filename(self) -> str:
        """Returns `<problem_name>.<algorithm_name>.<n_run>.pp.npz`."""
        return self.filename_prefix() + ".pp.npz"

    def result_filename(self) -> str:
        """Returns `<problem_name>.<algorithm_name>.<n_run>.csv`."""
        return self.filename_prefix() + ".csv"

    def top_layer_history_filename(self) -> str:
        """Returns the filename of the top layer history."""
        prefix = self.filename_prefix()
        name = self.problem_description["problem"]._name
        return f"{prefix}.1-{name}.npz"


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
                unspecified, defaults to `["igd"]`.
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

        self._output_dir_path = Path(output_dir_path)

        if performance_indicators is None:
            self._performance_indicators = ["igd"]
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

    def _all_pairs(self) -> List[Pair]:
        """Generate the list of all pairs to be run."""
        everything = product(
            self._algorithms.items(),
            self._problems.items(),
            range(1, self._n_runs + 1),
        )
        return [
            Pair(
                algorithm_description=aa,
                algorithm_name=an,
                n_run=r,
                problem_description=pp,
                problem_name=pn,
            )
            for (an, aa), (pn, pp), r in everything
        ]

    def _compute_performance_indicators(self) -> None:
        """
        Computes all performance indicators. It is assumed that all pairs have
        been ran, histories dumped, and that `_results` has been consolidated
        (see `_consolidate_pair_results`).
        """
        all_df = []

        for p in self._all_pairs():
            df = pd.DataFrame()

            # Load top layer history and the pareto history
            history_path = self._output_dir_path / (
                p.top_layer_history_filename()
            )
            pareto_history_path = self._output_dir_path / (
                p.pareto_population_filename()
            )
            if not (history_path.exists() and pareto_history_path.exists()):
                continue
            history = np.load(history_path)
            pareto_history = np.load(pareto_history_path)

            # Break down histories. Each element of this list is a tuple
            # containing the population's `X` and `F`, and the pareto
            # population's `X` and `F` at a given generation (or `_batch`).
            states = []
            for i in range(1, history["_batch"].max() + 1):
                hidx = history["_batch"] == i
                pidx = pareto_history["_batch"] == i
                states.append(
                    (
                        history["X"][hidx],
                        history["F"][hidx],
                        pareto_history["X"][pidx],
                        pareto_history["F"][pidx],
                    )
                )

            # Compute PIs
            # pylint: disable=cell-var-from-loop
            for pi in self._performance_indicators:
                logging.debug("Computing PI '%s' for pair [%s]", pi, p)
                f: Callable[
                    [np.ndarray, np.ndarray, np.ndarray, np.ndarray], float
                ] = lambda *_: np.nan
                if pi == "df":
                    problem = p.problem_description["problem"]
                    n_evals = p.problem_description.get("df_n_evals", 1)
                    delta_f = DeltaF(problem, n_evals)
                    f = lambda X, F, pX, pF: delta_f.do(F, X)
                elif pi in ["gd", "gd+", "igd", "igd+"]:
                    pf = p.problem_description.get(
                        "pareto_front",
                        self._global_pareto_population(
                            algorithm_name=p.algorithm_name,
                            problem_name=p.problem_name,
                        ).get("F"),
                    )
                    ind = get_performance_indicator(pi, pf)
                    f = lambda X, F, pX, pF: ind.do(F)
                elif pi == "hv" and "hv_ref_point" in p.problem_description:
                    hv = get_performance_indicator(
                        "hv", ref_point=p.problem_description["hv_ref_point"]
                    )
                    f = lambda X, F, pX, pF: hv.do(F)
                elif pi == "ps":
                    f = lambda X, F, pX, pF: pX.shape[0]

                df["perf_" + pi] = [f(*s) for s in states]

            df["algorithm"] = p.algorithm_name
            df["problem"] = p.problem_name
            df["n_run"] = p.n_run
            all_df.append(df)

        self._results = self._results.merge(
            pd.concat(all_df, ignore_index=True),
            how="outer",
            on=["algorithm", "problem", "n_run"],
        )

    def _consolidate_pair_results(self) -> None:
        """
        In `_run_pair`, if a pair is run successfully, it generates a CSV file
        `output_dir_path/<problem_name>.<algorithm_name>.<n_run>.csv`. This
        method consolidates all of them into a single dataframe and stores it
        in this benchmark's `_result` field.
        """
        logging.debug("Consolidating all pair statistics")
        all_df = []
        for pair in self._all_pairs():
            path = self._output_dir_path / pair.result_filename()
            if not path.exists():
                logging.debug(
                    "File %s does not exist. The corresponding pair most "
                    "likely didn't succeed",
                    path,
                )
                continue
            all_df.append(pd.read_csv(path))
        self._results = pd.concat(all_df, ignore_index=True)
        self._results["timedelta"] = pd.to_timedelta(
            self._results["timedelta"]
        )
        self._results = self._results.astype(
            {
                "algorithm": "category",
                "n_gen": "uint32",
                "n_run": "uint32",
                "problem": "category",
            }
        )

    @lru_cache(10)
    def _global_pareto_population(
        self,
        problem_name: str,
        algorithm_name: str,
    ) -> Population:
        """
        Given a problem-algorithm pair, loads and merges all pareto populations
        across all runs of that pair.
        """
        logging.debug(
            "Computing global Pareto population for pair %s - %s",
            problem_name,
            algorithm_name,
        )
        populations = []
        for n_run in range(1, self._n_runs + 1):
            data = np.load(
                self._output_dir_path
                / f"{problem_name}.{algorithm_name}.{n_run}.pp.npz"
            )
            population = Population.create(data["X"])
            population.set(
                F=data["F"], feasible=np.full((data["X"].shape[0], 1), True)
            )
            populations.append(population)
        return _merge_pareto_populations(populations)

    def _run_pair(
        self,
        pair: Pair,
    ) -> bool:
        """
        Runs a given algorithm against a given problem. See
        `nmoo.benchmark.Benchmark.run`. Immediately dumps the history of the
        problem and all wrapped problems with the following naming scheme:

            output_dir_path/<problem_name>.<algorithm_name>.<n_run>.<level>.npz

        where `level` is the depth of the wrapped problem, starting at `1`. See
        `nmoo.wrapped_problem.WrappedProblem.dump_all_histories`. It also dumps
        the compounded Pareto population for every at every generation (or just
        the last generation of `set_history` is set to `False` in the algorithm
        description) in

            output_dir_path/<problem_name>.<algorithm_name>.<n_run>.pp.npz

        Additionally, it generates a CSV file containing various statistics
        named:

            output_dir_path/<problem_name>.<algorithm_name>.<n_run>.csv

        The existence of this file is also used to determine if the pair has
        already been run when resuming a benchmark.

        Args: pair: A `Pair` object representint the problem-algorithm pair to
            run.

        Returns: Wether the run was successful or not.
        """
        result_file_path = self._output_dir_path / (pair.result_filename())
        if result_file_path.is_file():
            logging.debug("Pair [%s] has already been run, skipping.", pair)
            return True
        logging.info("Running pair [%s]", pair)

        pair.problem_description["problem"].start_new_run()
        evaluator = pair.problem_description.get(
            "evaluator",
            pair.algorithm_description.get("evaluator"),
        )
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
                save_history=True,
                seed=pair.algorithm_description.get("seed"),
                verbose=pair.algorithm_description.get("verbose", False),
            )
        except:  # pylint: disable=bare-except
            logging.error("Pair [%s] failed. Rescheduling...", pair)
            return False

        # Dump all layers histories
        if self._dump_histories:
            results.problem.dump_all_histories(
                self._output_dir_path,
                pair.filename_prefix(),
            )

        # Dump pareto sets
        pss = _consolidate_population_list([h.opt for h in results.history])
        np.savez_compressed(
            self._output_dir_path / pair.pareto_population_filename(), **pss
        )

        # Create and dump CSV file
        df = pd.DataFrame()
        df["n_gen"] = [a.n_gen for a in results.history]
        df["timedelta"] = results.algorithm.callback._deltas
        # Important to create these columns once the dataframe has its full
        # length
        df["algorithm"] = pair.algorithm_name
        df["problem"] = pair.problem_name
        df["n_run"] = pair.n_run
        df.to_csv(result_file_path, index=False)

        return True

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
        if not os.path.isdir(self._output_dir_path):
            os.mkdir(self._output_dir_path)
        pairs = self._all_pairs()
        executor = Parallel(n_jobs=n_jobs, **joblib_kwargs)
        current_round = 0
        while (
            self._max_retry < 0 or current_round <= self._max_retry
        ) and len(pairs) > 0:
            status = executor(
                delayed(Benchmark._run_pair)(self, pair) for pair in pairs
            )
            pairs = [p for p, s in zip(pairs, status) if not s]
            current_round += 1
        if pairs:
            logging.warning(
                "Benchmark finished, but some pairs could not be run "
                "successfully within the retry budget (%d):",
                self._max_retry,
            )
            for pair in pairs:
                logging.warning("    [%s]", pair)
        self._consolidate_pair_results()
        self._compute_performance_indicators()
        self.dump_results(self._output_dir_path / "benchmark.csv", index=False)


def _consolidate_population_list(populations: List[Population]) -> dict:
    """
    Transforms a list of populations into a dict containing the following

    * `X`: an `np.array` containing all `X` fields of all individuals across
      all populations;
    * `F`: an `np.array` containing all `F` fields of all individuals across
      all populations;
    * `G`: an `np.array` containing all `G` fields of all individuals across
      all populations;
    * `dF`: an `np.array` containing all `dF` fields of all individuals across
      all populations;
    * `dG`: an `np.array` containing all `dG` fields of all individuals across
      all populations;
    * `ddF`: an `np.array` containing all `ddF` fields of all individuals
      across all populations;
    * `ddG`: an `np.array` containing all `ddG` fields of all individuals
      across all populations;
    * `CV`: an `np.array` containing all `CV` fields of all individuals across
      all populations;
    * `feasible`: an `np.array` containing all `feasible` fields of all
      individuals across all populations;
    * `_batch`: the index of the population the individual belongs to.

    So all `np.arrays` have the same length, which is the total number of
    individual across all populations. Each "row" corresponds to the data
    associated to this individual (`X`, `F`, `G`, `dF`, `dG`, `ddF`, `ddG`,
    `CV`, `feasible`), as well as the population index it belongs to
    (`_batch`).
    """
    fields = ["X", "F", "G", "dF", "dG", "ddF", "ddG", "CV", "feasible"]
    data: Dict[str, List[np.ndarray]] = {f: [] for f in fields + ["_batch"]}
    for i, pop in enumerate(populations):
        for f in fields:
            data[f].append(pop.get(f))
        data["_batch"].append(np.full(len(pop), i + 1))
    return {k: np.concatenate(v) for k, v in data.items()}


def _merge_pareto_populations(populations: List[Population]) -> Population:
    """Simply merge population and apply `filter_optimum`."""
    population = Population.create()
    for p in populations:
        population = population.merge(population, p)
    result = filter_optimum(population)
    return result if result is not None else Population.create()
