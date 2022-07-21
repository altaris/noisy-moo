"""
Performance indicators plotting
"""
__docformat__ = "google"

from typing import Iterable, Optional

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
    x: str = "n_gen",
) -> sns.FacetGrid:
    """
    Plots all performance indicators in a grid of line plots.

    <center>
        <img src="https://github.com/altaris/noisy-moo/raw/main/imgs/plot_performance_indicators.png"
        alt="Example"/>
    </center>

    The columns of this grid correspond to the performance indicators, whereas
    the rows can be set to correspond to either `n_run`, `problem` or
    `algorithm`. For example, if `row="algorithm"` (as is the case above), then
    each row will correspond to an algorithm, whereas `n_run` and `problem`
    will be compounded in the line plots. If left to `None`, then `n_run`,
    `problem` and `algorithm` will all be compounded together.

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
        x (str): Column for the x axis. Should be among `n_gen` (the default),
            `n_eval`, or `timedelta`.
    """
    if algorithms is None:
        algorithms = benchmark._algorithms.keys()
    if performance_indicators is None:
        performance_indicators = benchmark._performance_indicators
    if problems is None:
        problems = benchmark._problems.keys()
    df = benchmark._results[
        ["algorithm", "n_eval", "n_gen", "n_run", "problem", "timedelta"]
        + ["perf_" + pi for pi in performance_indicators]
    ]
    df = df[(df.algorithm.isin(algorithms)) & (df.problem.isin(problems))]
    df.rename(
        columns={f"perf_{pi}": pi for pi in performance_indicators},
        inplace=True,
    )
    df = df.melt(
        id_vars=[
            c for c in df.columns if c not in benchmark._performance_indicators
        ],
        var_name="pi",
    )
    grid = sns.FacetGrid(df, col="pi", row=row, sharey=False)
    grid.map_dataframe(
        sns.lineplot, x=x, y="value", style="algorithm", hue="problem"
    )
    if legend:
        grid.add_legend()
    return grid
