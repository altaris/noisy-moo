"""
Performance indicators plotting
"""
__docformat__ = "google"

from typing import Iterable, Optional

import pandas as pd
import seaborn as sns

from nmoo.benchmark import Benchmark


# TODO: replace the benchmark argument by a benchmark_or_results_file_path, and
# retrieve the relevant benchmark specifications from the csv file.
def plot_performance_indicators(
    benchmark: Benchmark,
    row: Optional[str] = None,
    *,
    algorithms: Optional[Iterable[str]] = None,
    performance_indicators: Optional[Iterable[str]] = None,
    problems: Optional[Iterable[str]] = None,
    legend: bool = True,
) -> sns.FacetGrid:
    """
    Plots all performance indicators in a grid of line plots. The columns of
    this grid correspond to the performance indicators, whereas the rows can be
    set to correspond to either `n_run`, `problem` or `algorithm`. For example,
    if `row="problem"`, then each row will correspond to a problem, whereas
    `n_run` and `algorithm` will be compounded in the line plots. If left to
    `None`, then `n_run`, `problem` and `algorithm` will all be compounded
    together.

    Note:
        If you have the benchmark definition, the `benchmark.csv` file, but do
        not want to rerun the benchmark, you can use the following trick:

            benchmark = Benchmark(...)                               # Benchmark specification
            benchmark._results = pd.read_csv(path_to_benchmark_csv)  # Inject results
            plot_performance_indicators(benchmark, ...)              # Plot

    Args:
        benckmark: A (ran) benchmark object.
        row (Optional[str]): See above.
        algorithms (Optional[Iterable[str]]): List of algorithms to plot,
            defaults to all.
        performance_indicators (Optional[Iterable[str]]): List of performance
            indicators to plot, defaults to all.
        problems (Optional[Iterable[str]]): List of problems to plot, defaults
            to all.
        legend (bool): Wether to display the legend. Defaults to `True`.
    """
    if algorithms is None:
        algorithms = benchmark._algorithms.keys()
    if performance_indicators is None:
        performance_indicators = benchmark._performance_indicators
    if problems is None:
        problems = benchmark._problems.keys()
    results = benchmark._results
    results = results[
        (results.algorithm.isin(algorithms)) & (results.problem.isin(problems))
    ]
    all_tmp = []
    for p in performance_indicators:
        tmp = results[["algorithm", "problem", "n_run", "n_gen"]].copy()
        tmp["perf"], tmp["indicator"] = results["perf_" + p], p
        all_tmp.append(tmp)
    df = pd.concat(all_tmp, ignore_index=True)
    grid = sns.FacetGrid(df, col="indicator", row=row, sharey=False)
    grid.map_dataframe(
        sns.lineplot,
        x="n_gen",
        y="perf",
        style="algorithm",
        hue="problem",
    )
    if legend:
        grid.add_legend()
    return grid
