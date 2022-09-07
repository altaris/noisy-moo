# pylint: disable=too-many-lines

"""
A benchmarking utility. The following is perhaps the one of simplest benchmark
one could make
```py
from pymoo.problems.multi import ZDT1
from pymoo.algorithms.moo.nsga2 import NSGA2
from nmoo import Benchmark, GaussianNoise, KNNAvg, WrappedProblem

zdt1 = WrappedProblem(ZDT1())
noisy_zdt1 = GaussianNoise(zdt1, np.zeros(2), 1)
knnavg_zdt1 = KNNAvg(noisy_zdt1, max_distance=1.0)

benchmark = Benchmark(
    problems={
        "knnavg": {
            "problem": knnavg_zdt1
        }
    },
    algorithms={
        "nsga2": {
            "algorithm": NSGA2()
        }
    },
    n_runs=3,
    output_dir_path="./out",
)
```
which simply runs vanilla `NSGA2` against a KNN-Averaging-denoised
Gaussian-noised synthetic
[ZDT1](https://pymoo.org/problems/multi/zdt.html#ZDT1) against NSGA2[^nsga2], 3
times. The benchmark can be executed with
```
benchmark.run()
```
and the `./out` directory will be populated with various artefact, see below.

Refer to https://github.com/altaris/noisy-moo/blob/main/example.ipynb to get
started, or to https://github.com/altaris/noisy-moo/blob/main/example.py for a
more complete example.

## Artefact specification

After the benchmark above is run, the `./out` directory is populated with the
following artefacts:

* `benchmark.csv` the main result file. It has one row per (algorithm, problem,
  run number, generation). The columns are: `n_gen`, `n_eval`, `timedelta`,
  `algorithm`, `problem`, `n_run`, `perf_igd`. Here is a sample

        n_gen,n_eval,timedelta,algorithm,problem,n_run,perf_igd
        1,100,0 days 00:00:00.046010,nsga2,knnavg,1,2.7023936601855274
        2,200,0 days 00:00:00.110027,nsga2,knnavg,1,2.9920028540271617
        3,300,0 days 00:00:00.170194,nsga2,knnavg,1,2.808592743167947
        4,400,0 days 00:00:00.234336,nsga2,knnavg,1,2.7716447570482603
        5,500,0 days 00:00:00.300136,nsga2,knnavg,1,2.76605547730596
        6,600,0 days 00:00:00.367092,nsga2,knnavg,1,2.016998447316908
        7,700,0 days 00:00:00.432571,nsga2,knnavg,1,2.025674566580406
        8,800,0 days 00:00:00.501700,nsga2,knnavg,1,1.7875644431157067
        9,900,0 days 00:00:00.571355,nsga2,knnavg,1,2.5705921276809542

* `<problem>.<algorithm>.<run number>.csv`: same as `benchmark.csv` but only
  for a given (algorithm, problem, run number) triple.

* `<problem>.<algorithm>.<run number>.pi-<perf. indicator>.csv`: performance
  indicator file. Contains one row per generation. The columns are `perf_<perf.
  indicator name>`, `algorithm`, `problem`, `n_gen`, `n_run`. Here is a sample
  from `knnavg.nsga2.1.pi-igd.csv`:

        ,perf_igd,algorithm,problem,n_gen,n_run
        0,2.7023936601855274,nsga2,knnavg,1,1
        1,2.9920028540271617,nsga2,knnavg,2,1
        2,2.808592743167947,nsga2,knnavg,3,1
        3,2.7716447570482603,nsga2,knnavg,4,1
        4,2.76605547730596,nsga2,knnavg,5,1
        5,2.0169984473169076,nsga2,knnavg,6,1
        6,2.025674566580406,nsga2,knnavg,7,1
        7,1.7875644431157067,nsga2,knnavg,8,1
        8,2.5705921276809542,nsga2,knnavg,9,1
        9,2.245542743713137,nsga2,knnavg,10,1

* `<problem>.<algorithm>.<run number>.<layer number>-<layer name>.npz`: NPZ
  archive containing the history of all calls to a given layer of a given
  problem. In the example above, problem `knnavg_zda1` has three layers:
  `knn_avg` (layer 1, the outermost one), `gaussian_noise` (layer 2), and
  `wrapped_problem` (layer 3, the innermost one). Recall that you can set the
  name of a layer using the `name` argument in `WrappedProblem.__init__`. The
  keys are `X`, `F`, `_batch`, `_run`. It may also contain keys `G`, `dF`,
  `dG`, `ddF`, `ddG`, `CV`, `feasible` depending on the ground pymoo problem.
  The arrays at each keys have the same length (`shape[0]`), which is the
  number of individuals that have been evaluated throughout that run. In our
  example above, `out/knnavg.nsga2.1.1-knn_avg.npz` has keys `X`, `F`,
  `_batch`, `_run`, and the arrays have shape `(19600, 30)`, `(19600, 2)`,
  `(19600,)`, `(19600,)`, respectively. 30 is the number of variables of ZDT1,
  while 2 is the number of objectives.

* `<problem>.<algorithm>.<run number>.pp.npz`: Pareto population of a given
  (algorithm, problem, run number) triple. The keys are `X`, `F`, `G`, `dF`,
  `dG`, `ddF`, `ddG`, `CV`, `feasible`, `_batch`, and all arrays have the same
  length (`shape[0]`). Row `i` corresponds to an individual that was
  Pareto-ideal at generation `_batch[i]`.

* `<problem>.<algorithm>.gpp.npz`: *Global Pareto population* of a given
  problem-algorithm pair. It is the Pareto population of the population of all
  individuals designed across all runs and all generations of a given
  problem-algorithm pair. It is used to compute certain performance indicators
  in the absence of a baseline Pareto front. The keys are `X`, `F`, `G`, `dF`,
  `dG`, `ddF`, `ddG`, `CV`, `feasible`, `_batch`.

[^nsga2]: Deb, K., Agrawal, S., Pratap, A., Meyarivan, T. (2000). A Fast
    Elitist Non-dominated Sorting Genetic Algorithm for Multi-objective
    Optimization: NSGA-II. In: , et al. Parallel Problem Solving from Nature
    PPSN VI. PPSN 2000. Lecture Notes in Computer Science, vol 1917. Springer,
    Berlin, Heidelberg. https://doi.org/10.1007/3-540-45356-3_83

"""
__docformat__ = "google"

import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pymoo.factory import get_performance_indicator
from pymoo.optimize import minimize

from nmoo.callbacks import TimerCallback
from nmoo.indicators.delta_f import DeltaF
from nmoo.utils.population import pareto_frontier_mask, population_list_to_dict
from nmoo.denoisers import ResampleAverage

_PIC = Callable[[Dict[str, np.ndarray]], Optional[float]]
"""
Performance Indicator Callable. Type of a function that takes a state (dict of
`np.ndarray` with keys e.g. `F`, `X`, `pF`, etc.) and returns the value of a
performance indicator. See `Benchmark._compute_performance_indicator`.
"""


@dataclass
class PAPair:
    """
    Represents a problem-algorithm pair.
    """

    algorithm_description: Dict[str, Any]
    algorithm_name: str
    problem_description: Dict[str, Any]
    problem_name: str

    def __str__(self) -> str:
        return f"{self.problem_name} - {self.algorithm_name}"

    def global_pareto_population_filename(self) -> str:
        """Returns `<problem_name>.<algorithm_name>.gpp.npz`."""
        return f"{self.problem_name}.{self.algorithm_name}.gpp.npz"


@dataclass
class PARTriple(PAPair):
    """
    Represents a problem-algorithm-(run number) triple.
    """

    n_run: int

    def __str__(self) -> str:
        return f"{self.problem_name} - {self.algorithm_name} ({self.n_run})"

    def denoised_top_layer_history_filename(self) -> str:
        """Returns
        `<problem_name>.<algorithm_name>.<n_run>`.1-<top_layer_name>.denoised.npz`.
        """
        prefix = self.filename_prefix()
        name = self.problem_description["problem"]._name
        return f"{prefix}.1-{name}.denoised.npz"

    def filename_prefix(self) -> str:
        """Returns `<problem_name>.<algorithm_name>.<n_run>`."""
        return f"{self.problem_name}.{self.algorithm_name}.{self.n_run}"

    def innermost_layer_history_filename(self) -> str:
        """Returns the filename of the innermost layer history."""
        prefix = self.filename_prefix()
        problem = self.problem_description["problem"]
        inner = problem.innermost_wrapper()
        name, depth = inner._name, problem.depth()
        return f"{prefix}.{depth}-{name}.npz"

    def pareto_population_filename(self) -> str:
        """Returns `<problem_name>.<algorithm_name>.<n_run>.pp.npz`."""
        return self.filename_prefix() + ".pp.npz"

    def pi_filename(self, pi_name: str) -> str:
        """
        Returns `<problem_name>.<algorithm_name>.<n_run>.pi-<pi_name>.csv`.
        """
        return self.filename_prefix() + f".pi-{pi_name}.csv"

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
        "ggd",
        "ggd+",
        "ghv",
        "gigd",
        "gigd+",
        "hv",
        "igd",
        "igd+",
        "ps",
        "rggd",
        "rggd+",
        "rghv",
        "rgigd",
        "rgigd+",
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
    Maximum number of attempts to run a given problem-algorithm-(run number)
    triple before giving up.
    """

    _n_runs: int
    """
    Number of times to run a given problem-algorithm pair.
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

    _seeds: List[Optional[int]]
    """
    List of seeds to use. Must be of length `_n_runs`.
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
        seeds: Optional[List[Optional[int]]] = None,
    ):
        """
        Constructor. The set of problems to be benchmarked is represented by a
        dictionary with the following structure:

            problems = {
                <problem_name>: <problem_description>,
                <problem_name>: <problem_description>,
            }

        where `<problem_name>` is a user-defined string (but stay reasonable
        since it may be used in filenames), and `<problem_description>` is a
        dictionary with the following keys: * `df_n_evals` (int, optional): see
        the explanation of the `df`
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
        dictionary with the following keys: * `algorithm`: a pymoo `Algorithm`
        object; note that it is deepcopied
            for every run of `minimize`;
        * `display` (optional): a custom `pymoo.util.display.Display` object
            for customization purposes;
        * `evaluator` (optional): an algorithm evaluator object; note that it
            is deepcopied for every run of `minimize`;
        * `return_least_infeasible` (optional, bool): if the algorithm cannot
            find a feasible solution, wether the least infeasible solution
            should still be returned; defaults to `False`;
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
                problem-algorithm-(run number) triple before giving up. Set it
                to `-1` to retry indefinitely.
            n_runs (int): Number of times to run a given problem-algorithm
                pair.
            problems (Dict[str, dict]): Dict of all problems to be benchmarked.
            performance_indicators (Optional[List[str]]): List of perfomance
                indicators to be calculated and included in the result
                dataframe (see `Benchmark.final_results`). Supported indicators
                are
                * `df`: ΔF metric, see the documentation of
                  `nmoo.indicators.delta_f.DeltaF`; `df_n_eval` should be set
                  in the problem description, but it default to 1 if not;
                * `gd`: [generational
                  distance](https://pymoo.org/misc/indicators.html#Generational-Distance-(GD)),
                  requires `pareto_front` to be set in the problem description
                  dictionaries, otherwise the value of this indicator will be
                  `NaN`;
                * `gd+`: [generational distance
                  plus](https://pymoo.org/misc/indicators.html#Generational-Distance-Plus-(GD+)),
                  requires `pareto_front` to be set in the problem description
                  dictionaries, otherwise the value of this indicator will be
                  `NaN`;
                * `hv`:
                  [hypervolume](https://pymoo.org/misc/indicators.html#Hypervolume),
                  requires `hv_ref_point` to be set in the problem discription
                  dictionaries, otherwise the value of this indicator will be
                  `NaN`;
                * `igd`: [inverted generational
                  distance](https://pymoo.org/misc/indicators.html#Inverted-Generational-Distance-(IGD)),
                  requires `pareto_front` to be set in the problem description
                  dictionaries, otherwise the value of this indicator will be
                  `NaN`;
                * `igd+`: [inverted generational distance
                  plus](https://pymoo.org/misc/indicators.html#Inverted-Generational-Distance-Plus-(IGD+)),
                  requires `pareto_front` to be set in the problem description
                  dictionaries, otherwise the value of this indicator will be
                  `NaN`;
                * `ps`: population size, or equivalently, the size of the
                  current Pareto front;
                * `ggd`: ground generational distance, where the ground
                  problem's predicted objective values are used instead of the
                  outer problem's; requires `pareto_front` to be set in the
                  problem description dictionaries, otherwise the value of this
                  indicator will be `NaN`;
                * `ggd+`: ground generational distance plus; requires
                  `pareto_front` to be set in the problem description
                  dictionaries, otherwise the value of this indicator will be
                  `NaN`;
                * `ghv`: ground hypervolume; requires `hv_ref_point` to be set
                  in the problem discription dictionaries, otherwise the value
                  of this indicator will be `NaN`;
                * `gigd`: ground inverted generational distance; requires
                  `pareto_front` to be set in the problem description
                  dictionaries, otherwise the value of this indicator will be
                  `NaN`;
                * `gigd+`: ground inverted generational distance plus; requires
                  `pareto_front` to be set in the problem description
                  dictionaries, otherwise the value of this indicator will be
                  `NaN`.
                * `rggd`: resampled ground generational distance, where the
                  ground problem's predicted objective values (resampled and
                  averaged a given number of times) are used instead of the
                  outer problem's; requires `pareto_front` to be set in the
                  problem description dictionaries, otherwise the value of this
                  indicator will be `NaN`; `rg_n_eval` should also be set in
                  the problem description, but defaults to 1 if not;
                * `rggd+`: resampled ground generational distance plus;
                  requires `pareto_front` to be set in the problem description
                  dictionaries, otherwise the value of this indicator will be
                  `NaN`; `rg_n_eval` should also be set in the problem
                  description, but defaults to 1 if not; `rg_n_eval` should
                  also be set in the problem description, but defaults to 1 if
                  not;
                * `rghv`: resampled ground hypervolume; requires `hv_ref_point`
                  to be set in the problem discription dictionaries, otherwise
                  the value of this indicator will be `NaN`; `rg_n_eval` should
                  also be set in the problem description, but defaults to 1 if
                  not;
                * `rgigd`: resampled ground inverted generational distance;
                  requires `pareto_front` to be set in the problem description
                  dictionaries, otherwise the value of this indicator will be
                  `NaN`; `rg_n_eval` should also be set in the problem
                  description, but defaults to 1 if not;
                * `rgigd+`: resampled ground inverted generational distance
                  plus; requires `pareto_front` to be set in the problem
                  description dictionaries, otherwise the value of this
                  indicator will be `NaN`; `rg_n_eval` should also be set in
                  the problem description, but defaults to 1 if not.

                In the result dataframe, the corresponding columns will be
                named `perf_<name of indicator>`, e.g. `perf_igd`. If left
                unspecified, defaults to `["igd"]`.

            seeds (Optional[List[Optional[int]]]): List of seeds to use. The
                first seed will be used for the first run of every
                algorithm-problem pair, etc.
        """
        self._output_dir_path = Path(output_dir_path)
        self._set_problems(problems)
        self._set_algorithms(algorithms)
        if n_runs <= 0:
            raise ValueError(
                "The number of run (for each problem-algorithm pair) must be "
                "at least 1."
            )
        self._n_runs = n_runs
        self._dump_histories = dump_histories
        self._set_performance_indicators(performance_indicators)
        self._max_retry = max_retry
        if seeds is None:
            self._seeds = [None] * n_runs
        elif len(seeds) < n_runs:
            raise ValueError(
                f"Not enough seeds: provided {len(seeds)} seeds but specified "
                f"{n_runs} runs."
            )
        else:
            if len(seeds) > n_runs:
                logging.warning(
                    "Too many seeds: provided %d but only need %d "
                    "(i.e. n_run)",
                    len(seeds),
                    n_runs,
                )
            self._seeds = seeds

    def _compute_global_pareto_population(self, pair: PAPair) -> None:
        """
        Computes the global Pareto population of a given problem-algorithm
        pair. See `compute_global_pareto_populations`. Assumes that the global
        Pareto population has not already been calculated, i.e. that
        `<output_dir_path>/<problem>.<algorithm>.gpp.npz` does not exist.
        """
        logging.debug("Computing global Pareto population for pair [%s]", pair)
        gpp_path = (
            self._output_dir_path / pair.global_pareto_population_filename()
        )
        populations: Dict[str, List[np.ndarray]] = {}
        for n_run in range(1, self._n_runs + 1):
            triple = PARTriple(
                algorithm_description=pair.algorithm_description,
                algorithm_name=pair.algorithm_name,
                n_run=n_run,
                problem_description=pair.problem_description,
                problem_name=pair.problem_name,
            )
            path = self._output_dir_path / triple.pareto_population_filename()
            if not path.exists():
                logging.debug(
                    "File %s does not exist. The corresponding triple [%s] "
                    "most likely hasn't finished or failed",
                    path,
                    triple,
                )
                continue
            data = np.load(path, allow_pickle=True)
            for k, v in data.items():
                populations[k] = populations.get(k, []) + [v]
            data.close()

        consolidated = {k: np.concatenate(v) for k, v in populations.items()}
        if "F" not in consolidated:
            logging.error(
                "No Pareto population file found for pair [%s]. This is "
                "most likely because none of the runs finished or succeeded.",
                pair,
            )
            return
        mask = pareto_frontier_mask(consolidated["F"])
        np.savez_compressed(
            gpp_path,
            **{k: v[mask] for k, v in consolidated.items()},
        )

    # pylint: disable=too-many-branches
    def _compute_performance_indicator(
        self, triple: PARTriple, pi_name: str
    ) -> None:
        """
        Computes a performance indicators for a given problem-algorithm-(run
        number) triple and stores it under
        `<problem_name>.<algorithm_name>.<n_run>.pi-<pi_name>.csv`. Assumes
        that the this performance indicator has not already been calculated,
        i.e. that that file does not exist.

        Warning:
            This fails if either the top layer history or the pareto population
            artefact (`<problem_name>.<algorithm_name>.<n_run>.pp.npz`) could
            not be loaded as numpy arrays.

        Todo:
            Refactor (again)
        """
        logging.debug("Computing PI '%s' for triple [%s]", pi_name, triple)

        pic: _PIC = lambda _: np.nan
        if pi_name == "df":
            problem = triple.problem_description["problem"]
            n_evals = triple.problem_description.get("df_n_evals", 1)
            delta_f = DeltaF(problem, n_evals)
            pic = lambda s: delta_f.do(s["F"], s["X"])
        elif pi_name in ["gd", "gd+", "igd", "igd+"]:
            pic = self._get_pic_gd_type(triple, pi_name)
        elif pi_name in ["ggd", "ggd+", "gigd", "gigd+"]:
            pic = self._get_pic_gd_type(triple, pi_name[1:])
        elif pi_name in ["rggd", "rggd+", "rgigd", "rgigd+"]:
            pic = self._get_pic_gd_type(triple, pi_name[2:])
        elif (
            pi_name in ["hv", "ghv", "rghv"]
            and "hv_ref_point" in triple.problem_description
        ):
            ref_point = triple.problem_description["hv_ref_point"]
            pi = get_performance_indicator("hv", ref_point=ref_point)
            pic = lambda s: pi.do(s["F"])
        elif pi_name == "ps":
            pic = lambda s: s["X"].shape[0]
        else:
            logging.warning(
                "Unprocessable performance indicator '%s'. This could be "
                "because required data is missing.",
                pi_name,
            )

        # On which history is the PIC going to be called? By default, it is on
        # the top layer history.
        if pi_name in ["ps"]:
            history = np.load(
                self._output_dir_path / triple.pareto_population_filename()
            )
        elif pi_name in ["ggd", "ggd+", "ghv", "gigd", "gigd+"]:
            history = np.load(
                self._output_dir_path
                / triple.innermost_layer_history_filename()
            )
        elif pi_name in ["rggd", "rggd+", "rghv", "rgigd", "rgigd+"]:
            history = self._get_rg_history(triple)
        else:
            history = np.load(
                self._output_dir_path / triple.top_layer_history_filename()
            )

        states: List[Dict[str, np.ndarray]] = []
        for i in range(1, history["_batch"].max() + 1):
            idx = history["_batch"] == i
            states.append({"X": history["X"][idx], "F": history["F"][idx]})
        df = pd.DataFrame()
        df["perf_" + pi_name] = list(map(pic, states))
        df["algorithm"] = triple.algorithm_name
        df["problem"] = triple.problem_name
        df["n_gen"] = range(1, len(states) + 1)
        df["n_run"] = triple.n_run
        df.to_csv(self._output_dir_path / triple.pi_filename(pi_name))

        if pi_name == "df":
            delta_f.dump_history(
                self._output_dir_path
                / triple.denoised_top_layer_history_filename()
            )

    def _get_pic_gd_type(self, triple: PARTriple, pi_name: str) -> _PIC:
        """
        Returns the `_PIC` corresponding to the either the `gd`, `gd+`, `igd`,
        or `igd+` performance indicator. As a reminder, a `_PIC`, or
        Performance Indicator Callable, is a function that takes a dict of
        `np.ndarray` and returns an optional `float`. In this case, the dict
        must have the key `F`.
        """
        if "pareto_front" in triple.problem_description:
            pf = triple.problem_description.get("pareto_front")
        else:
            path = (
                self._output_dir_path
                / triple.global_pareto_population_filename()
            )
            data = np.load(path)
            pf = data["F"]
            data.close()
        pi = get_performance_indicator(pi_name, pf)
        return lambda s: pi.do(s["F"])

    def _get_rg_history(self, triple: PARTriple) -> Dict[str, np.ndarray]:
        """
        Returns the `X` and `F` history of the ground problem of the triple,
        but where `F` has been resampled a given number of times (`rg_n_evals`
        parameter in the problem's description). This involves wrapping the
        ground problem in a `nmoo.denoisers.ResampleAverage` and evaluating the
        history's `X` array.
        """
        history = dict(
            np.load(
                self._output_dir_path / triple.top_layer_history_filename()
            )
        )
        rgp = ResampleAverage(
            triple.problem_description["problem"].ground_problem(),
            triple.problem_description.get("rg_n_eval", 1),
        )
        history["F"] = rgp.evaluate(history["X"], return_values_of="F")
        return history

    def _par_triple_done(self, triple: PARTriple) -> bool:
        """
        Wether a problem-algorithm-(run number) has been successfully executed.
        This is determined by checking if
        `_output_dir_path/triple.result_filename()` exists or not.
        """
        return (self._output_dir_path / triple.result_filename()).is_file()

    def _run_par_triple(
        self,
        triple: PARTriple,
    ) -> None:
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

        The existence of this file is also used to determine if the
        problem-algorithm-(run number) triple has already been run when
        resuming a benchmark.

        Args:
            triple: A `PARTriple` object representing the
                problem-algorithm-(run number) triple to run.
        """
        logging.info("Running triple [%s]", triple)

        triple.problem_description["problem"].start_new_run()
        evaluator = triple.problem_description.get(
            "evaluator",
            triple.algorithm_description.get("evaluator"),
        )
        try:
            seed = self._seeds[triple.n_run - 1]
            problem = deepcopy(triple.problem_description["problem"])
            problem.reseed(seed)
            results = minimize(
                problem,
                triple.algorithm_description["algorithm"],
                termination=triple.algorithm_description.get("termination"),
                copy_algorithm=True,
                copy_termination=True,
                # extra Algorithm.setup kwargs
                callback=TimerCallback(),
                display=triple.algorithm_description.get("display"),
                evaluator=deepcopy(evaluator),
                return_least_infeasible=triple.algorithm_description.get(
                    "return_least_infeasible", False
                ),
                save_history=True,
                seed=seed,
                verbose=triple.algorithm_description.get("verbose", False),
            )
        except Exception as e:  # pylint: disable=broad-except
            logging.error("Triple [%s] failed: %s", triple, e)
            return

        # Dump all layers histories
        if self._dump_histories:
            results.problem.dump_all_histories(
                self._output_dir_path,
                triple.filename_prefix(),
            )

        # Dump pareto sets
        np.savez_compressed(
            self._output_dir_path / triple.pareto_population_filename(),
            **population_list_to_dict([h.opt for h in results.history]),
        )

        # Create and dump CSV file
        df = pd.DataFrame()
        df["n_gen"] = [a.n_gen for a in results.history]
        df["n_eval"] = [a.evaluator.n_eval for a in results.history]
        df["timedelta"] = results.algorithm.callback._deltas
        # Important to create these columns once the dataframe has its full
        # length
        df["algorithm"] = triple.algorithm_name
        df["problem"] = triple.problem_name
        df["n_run"] = triple.n_run
        df.to_csv(
            self._output_dir_path / triple.result_filename(),
            index=False,
        )

    def _set_algorithms(self, algorithms: Dict[str, dict]) -> None:
        """Validates and sets the algorithms dict"""
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

    def _set_performance_indicators(
        self, performance_indicators: Optional[List[str]]
    ) -> None:
        """Validates and sets the performance indicator list"""
        if performance_indicators is None:
            self._performance_indicators = ["igd"]
        else:
            self._performance_indicators = []
            for pi in set(performance_indicators):
                if pi not in Benchmark.SUPPORTED_PERFOMANCE_INDICATORS:
                    raise ValueError(f"Unknown performance indicator '{pi}'")
                self._performance_indicators.append(pi)
            self._performance_indicators = sorted(self._performance_indicators)

    def _set_problems(self, problems: Dict[str, dict]) -> None:
        """Validates and sets the problem dict"""
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

    def all_pa_pairs(self) -> List[PAPair]:
        """Generate the list of all problem-algorithm pairs."""
        everything = product(
            self._algorithms.items(),
            self._problems.items(),
        )
        return [
            PAPair(
                algorithm_description=aa,
                algorithm_name=an,
                problem_description=pp,
                problem_name=pn,
            )
            for (an, aa), (pn, pp) in everything
        ]

    def all_par_triples(self) -> List[PARTriple]:
        """Generate the list of all problem-algorithm-(run number) triples."""
        everything = product(
            self._algorithms.items(),
            self._problems.items(),
            range(1, self._n_runs + 1),
        )
        return [
            PARTriple(
                algorithm_description=aa,
                algorithm_name=an,
                n_run=r,
                problem_description=pp,
                problem_name=pn,
            )
            for (an, aa), (pn, pp), r in everything
        ]

    def compute_global_pareto_populations(
        self, n_jobs: int = -1, **joblib_kwargs
    ) -> None:
        """
        The global Pareto population of a problem-algorithm pair is the merged
        population of all pareto populations across all runs of that pair. This
        function calculates global Pareto population of all pairs and dumps it
        to `<output_dir_path>/<problem>.<algorithm>.gpp.npz`. If that file
        exists for a given problem-algorithm pair, then the global Pareto
        population (of that pair) is not recalculated.
        """
        logging.info("Computing global Pareto populations")
        executor = Parallel(n_jobs=n_jobs, **joblib_kwargs)
        executor(
            delayed(Benchmark._compute_global_pareto_population)(self, p)
            for p in self.all_pa_pairs()
            if not (
                self._output_dir_path / p.global_pareto_population_filename()
            ).is_file()
        )

    def compute_performance_indicators(
        self, n_jobs: int = -1, **joblib_kwargs
    ) -> None:
        """
        Computes all performance indicators and saves the corresponding
        dataframes in
        `output_path/<problem_name>.<algorithm_name>.<n_run>.pi-<pi_name>.csv`.
        If that file exists for a given problem-algorithm-(run
        number)-(performance indicator) tuple, then it is not recalculated.
        """
        logging.info("Computing performance indicators")
        everything = product(
            self.all_par_triples(), self._performance_indicators
        )
        executor = Parallel(n_jobs=n_jobs, **joblib_kwargs)
        executor(
            delayed(Benchmark._compute_performance_indicator)(self, t, pi)
            for t, pi in everything
            if not (self._output_dir_path / t.pi_filename(pi)).is_file()
        )

    def consolidate(self) -> None:
        """
        Merges all statistics dataframes
        (`<problem_name>.<algorithm_name>.<n_run>.csv`) and all PI dataframes
        (`<problem_name>.<algorithm_name>.<n_run>.pi.csv`) into a single
        dataframe, and saves it under `output_dir_path/benchmark.csv`.
        """
        logging.info("Consolidating statistics")
        all_df = []
        for triple in self.all_par_triples():
            path = self._output_dir_path / triple.result_filename()
            if not path.exists():
                logging.debug(
                    "Statistic file %s does not exist. The corresponding "
                    "triple [%s] most likely hasn't finished or failed",
                    path,
                    triple,
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

        logging.info("Consolidating performance indicators")
        all_df = []
        for triple in self.all_par_triples():
            df = pd.DataFrame()
            for pi_name in self._performance_indicators:
                path = self._output_dir_path / triple.pi_filename(pi_name)
                if not path.exists():
                    logging.debug("PI file %s does not exist.", path)
                    continue
                tmp = pd.read_csv(path)
                if df.empty:
                    df = tmp
                else:
                    col = "perf_" + pi_name
                    df[col] = tmp[col]
            all_df.append(df)

        self._results = self._results.merge(
            pd.concat(all_df, ignore_index=True),
            how="outer",
            on=["algorithm", "problem", "n_gen", "n_run"],
        )

        # ???
        if "Unnamed: 0" in self._results:
            del self._results["Unnamed: 0"]

        path = self._output_dir_path / "benchmark.csv"
        logging.info("Writing results to %s", path)
        self.dump_results(path, index=False)

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

    def run(
        self,
        n_jobs: int = -1,
        n_post_processing_jobs: int = 2,
        **joblib_kwargs,
    ):
        """
        Runs the benchmark sequentially. Makes your laptop go brr. The
        histories of all problems are progressively dumped in the specified
        output directory as the benchmark run. At the end, the benchmark
        results are dumped in `output_dir_path/benchmark.csv`.

        Args:
            n_jobs (int): Number of processes to use. See the `joblib.Parallel`_
                documentation. Defaults to `-1`, i.e. all CPUs are used.
            n_post_processing_jobs (int): Number of processes to use for post
                processing tasks (computing global Pareto populations and
                performance indicators). These are memory-intensive tasks.
                Defaults to `2`.
            joblib_kwargs (dict): Additional kwargs to pass on to the
                `joblib.Parallel`_ instance.

        .. _joblib.Parallel:
            https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
        """
        if not os.path.isdir(self._output_dir_path):
            os.mkdir(self._output_dir_path)
        triples = self.all_par_triples()
        executor = Parallel(n_jobs=n_jobs, **joblib_kwargs)
        current_round = 0
        while (
            self._max_retry < 0 or current_round <= self._max_retry
        ) and any(not self._par_triple_done(t) for t in triples):
            executor(
                delayed(Benchmark._run_par_triple)(self, t)
                for t in triples
                if not self._par_triple_done(t)
            )
            current_round += 1
        if any(not self._par_triple_done(t) for t in triples):
            logging.warning(
                "Benchmark finished, but some triples could not be run "
                "successfully within the retry budget (%d):",
                self._max_retry,
            )
            for t in filter(lambda x: not self._par_triple_done(x), triples):
                logging.warning("    [%s]", t)
        self.compute_global_pareto_populations(
            n_post_processing_jobs, **joblib_kwargs
        )
        self.compute_performance_indicators(
            n_post_processing_jobs, **joblib_kwargs
        )
        self.consolidate()
