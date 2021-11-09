"""
noisy-moo CLI
"""
__docformat__ = "google"

import logging
import os
import re
import sys
from importlib import import_module
from itertools import product
from pathlib import Path
from typing import Optional

import click

try:
    import nmoo
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).absolute().parent.parent))
    import nmoo

sys.path.append(os.getcwd())


def _apply_overrides(
    benchmark: nmoo.benchmark.Benchmark,
    n_runs: Optional[int] = None,
    only_problems: str = "",
    exclude_problems: str = "$",
    only_algorithms: str = "",
    exclude_algorithms: str = "$",
    output_dir: Optional[Path] = None,
) -> None:
    """Applies overrides to a benchmark."""
    if n_runs is not None:
        benchmark._n_runs = n_runs
        logging.info("Overridden n_runs to %d", n_runs)
    if output_dir is not None:
        benchmark._output_dir_path = output_dir
        logging.info("Overridden output directory to %s", output_dir)
    if _include_exclude(benchmark._problems, only_problems, exclude_problems):
        logging.info(
            "Overridden problem list to %s",
            list(benchmark._problems.keys()),
        )
    if _include_exclude(
        benchmark._algorithms, only_algorithms, exclude_algorithms
    ):
        logging.info(
            "Overridden algorithm list to %s",
            list(benchmark._algorithms.keys()),
        )


def _include_exclude(
    dictionary: dict,
    include_pattern: str,
    exclude_pattern: str,
) -> bool:
    """
    Filters the items of a dictionary based on a include / exclude regexp pair.
    Returns `True` if the size of the dictionary changed.
    """
    incl, excl = re.compile(include_pattern), re.compile(exclude_pattern)
    keys = list(dictionary.keys())
    for k in keys:
        if excl.match(k) or not incl.match(k):
            del dictionary[k]
    return len(dictionary) != len(keys)


def _get_benchmark(path: str) -> nmoo.benchmark.Benchmark:
    """
    From a function "path" of the form `module[.submodule...]:function`,
    imports `module[.submodule...]` and returns `function`.
    """
    try:
        module_name, function_name = path.split(":")
        module = import_module(module_name)
        factory = getattr(module, function_name)
        benchmark = factory()
        assert isinstance(benchmark, nmoo.benchmark.Benchmark)
        return benchmark
    except AssertionError:
        logging.fatal("Factory '%s' did not return a Benchmark object.", path)
    except AttributeError:
        logging.fatal(
            "Module '%s' has no attribute '%s'", module_name, function_name
        )
    except ModuleNotFoundError as e:
        logging.fatal("%s", e)
    except TypeError:
        logging.fatal("Factory '%s' is not callable.", function_name)
    sys.exit(-1)


@click.group()
@click.option(
    "--logging-level",
    help=(
        "Logging level, among 'debug', 'info', 'warning', 'error', and "
        "'critical'."
    ),
    type=click.STRING,
)
def main(logging_level: str) -> None:
    """noisy-moo CLI"""
    logging_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    logging.basicConfig(
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        level=logging_levels.get(logging_level, logging.INFO),
    )


@main.command()
@click.argument(
    "benchmark",
    type=click.STRING,
)
@click.option(
    "--exclude-algorithms",
    default="$",
    help=(
        "Overrides the benchmark's algorithm list. Algorithms whose name "
        "matches the specified regexp will be excluded. Can be used in "
        "combination with --only-algorithms."
    ),
    type=click.STRING,
)
@click.option(
    "--exclude-problems",
    default="$",
    help=(
        "Overrides the benchmark's problem list. Problems whose name "
        "matches the specified regexp will be excluded. Can be used in "
        "combination with --only-problems."
    ),
    type=click.STRING,
)
@click.option(
    "--n-runs",
    help="Overrides the benchmark's 'n_runs' attribute.",
    type=click.INT,
)
@click.option(
    "--only-algorithms",
    default="",
    help=(
        "Overrides the benchmark's algorithm list. Only the algorithms whose "
        "name matches the specified regexp will be considered. Can be used in "
        "combination with --exclude-algorithms."
    ),
    type=click.STRING,
)
@click.option(
    "--only-problems",
    default="",
    help=(
        "Overrides the benchmark's problem list. Only the problems whose name "
        "matches the specified regexp will be considered. Can be used in "
        "combination with --exclude-problems."
    ),
    type=click.STRING,
)
def consolidate(
    benchmark: str,
    n_runs: Optional[int],
    only_problems: str,
    exclude_problems: str,
    only_algorithms: str,
    exclude_algorithms: str,
):
    """
    Consolidates the benchmark with the data calculated so far into
    "benchmark.csv". Can be safely called while the benchmark is running.
    """
    b = _get_benchmark(benchmark)
    _apply_overrides(
        b,
        n_runs=n_runs,
        only_problems=only_problems,
        exclude_problems=exclude_problems,
        only_algorithms=only_algorithms,
        exclude_algorithms=exclude_algorithms,
    )
    b.consolidate()


@main.command()
@click.argument(
    "benchmark",
    type=click.STRING,
)
@click.option(
    "--exclude-algorithms",
    default="$",
    help=(
        "Overrides the benchmark's algorithm list. Algorithms whose name "
        "matches the specified regexp will be excluded. Can be used in "
        "combination with --only-algorithms."
    ),
    type=click.STRING,
)
@click.option(
    "--exclude-problems",
    default="$",
    help=(
        "Overrides the benchmark's problem list. Problems whose name "
        "matches the specified regexp will be excluded. Can be used in "
        "combination with --only-problems."
    ),
    type=click.STRING,
)
@click.option(
    "--n-jobs",
    default=-1,
    help="Number of benchmark jobs.",
    type=click.INT,
)
@click.option(
    "--n-post-processing-jobs",
    default=-1,
    help="Number of post-processing jobs.",
    type=click.INT,
)
@click.option(
    "--n-runs",
    help="Overrides the benchmark's 'n_runs' attribute.",
    type=click.INT,
)
@click.option(
    "--only-algorithms",
    default="",
    help=(
        "Overrides the benchmark's algorithm list. Only the algorithms whose "
        "name matches the specified regexp will be considered. Can be used in "
        "combination with --exclude-algorithms."
    ),
    type=click.STRING,
)
@click.option(
    "--only-problems",
    default="",
    help=(
        "Overrides the benchmark's problem list. Only the problems whose name "
        "matches the specified regexp will be considered. Can be used in "
        "combination with --exclude-problems."
    ),
    type=click.STRING,
)
@click.option(
    "--output-dir",
    help="Overrides the benchmark's output directory.",
    type=click.Path(
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        path_type=Path,
    ),
)
@click.option(
    "--restart-on-crash/--no-restart-on-crash",
    help=(
        "Restarts the benchmark if it crashes. This can potentially lead an "
        "infinite loop if the benchmark keeps crashing."
    ),
    default=False,
)
@click.option(
    "--verbose",
    default=0,
    help="Joblib's verbosity level.",
    type=click.INT,
)
def run(
    benchmark: str,
    n_jobs: int,
    n_post_processing_jobs: int,
    verbose: int,
    n_runs: Optional[int],
    only_problems: str,
    exclude_problems: str,
    only_algorithms: str,
    exclude_algorithms: str,
    output_dir: Optional[Path],
    restart_on_crash: bool,
) -> None:
    """
    Runs a benchmark.

    Imports and executes a BENCHMARK, which is a string of the form
    'module[.submodule...]:function'. The 'function' returns the actual
    Benchmark object, and should be callable without any argument.
    """
    b = _get_benchmark(benchmark)
    _apply_overrides(
        b,
        n_runs=n_runs,
        only_problems=only_problems,
        exclude_problems=exclude_problems,
        only_algorithms=only_algorithms,
        exclude_algorithms=exclude_algorithms,
        output_dir=output_dir,
    )
    restart = True
    while restart:
        try:
            b.run(
                n_jobs=n_jobs,
                n_post_processing_jobs=n_post_processing_jobs,
                verbose=verbose,
            )
        except KeyboardInterrupt:
            restart = False
        except Exception as e:  # pylint: disable=broad-except
            logging.error("Benchmark crashed: %s", e)
            if restart_on_crash:
                restart = True
            else:
                raise
        else:
            restart = False


@main.command()
@click.argument(
    "benchmark",
    type=click.STRING,
)
@click.option(
    "--exclude-algorithms",
    default="$",
    help=(
        "Overrides the benchmark's algorithm list. Algorithms whose name "
        "matches the specified regexp will be excluded. Can be used in "
        "combination with --only-algorithms."
    ),
    type=click.STRING,
)
@click.option(
    "--exclude-problems",
    default="$",
    help=(
        "Overrides the benchmark's problem list. Problems whose name "
        "matches the specified regexp will be excluded. Can be used in "
        "combination with --only-problems."
    ),
    type=click.STRING,
)
@click.option(
    "--n-runs",
    help="Overrides the benchmark's 'n_runs' attribute.",
    type=click.INT,
)
@click.option(
    "--only-algorithms",
    default="",
    help=(
        "Overrides the benchmark's algorithm list. Only the algorithms whose "
        "name matches the specified regexp will be considered. Can be used in "
        "combination with --exclude-algorithms."
    ),
    type=click.STRING,
)
@click.option(
    "--only-problems",
    default="",
    help=(
        "Overrides the benchmark's problem list. Only the problems whose name "
        "matches the specified regexp will be considered. Can be used in "
        "combination with --exclude-problems."
    ),
    type=click.STRING,
)
@click.option(
    "--output-dir",
    help="Overrides the benchmark's output directory.",
    type=click.Path(
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        path_type=Path,
    ),
)
def tally(
    benchmark: str,
    n_runs: Optional[int],
    only_problems: str,
    exclude_problems: str,
    only_algorithms: str,
    exclude_algorithms: str,
    output_dir: Optional[Path],
):
    """
    Reports the current completion of the benchmark. Can be safely called while
    the benchmark is running.
    """
    b = _get_benchmark(benchmark)
    _apply_overrides(
        b,
        n_runs=n_runs,
        only_problems=only_problems,
        exclude_problems=exclude_problems,
        only_algorithms=only_algorithms,
        exclude_algorithms=exclude_algorithms,
        output_dir=output_dir,
    )
    all_triples = b.all_par_triples()
    all_gpps = {t.global_pareto_population_filename() for t in all_triples}
    all_pis = list(product(all_triples, b._performance_indicators))
    n_run = sum(
        map(
            lambda t: int(
                (b._output_dir_path / t.result_filename()).is_file()
            ),
            all_triples,
        )
    )
    n_gpp = sum(
        map(lambda p: int((b._output_dir_path / p).is_file()), all_gpps)
    )
    n_pi = sum(
        map(
            lambda x: int(
                (b._output_dir_path / x[0].pi_filename(x[1])).is_file()
            ),
            all_pis,
        )
    )
    logging.info("Runs: %d/%d", n_run, len(all_triples))
    logging.info("GPPs: %d/%d", n_gpp, len(all_gpps))
    logging.info("PIs: %d/%d", n_pi, len(all_pis))


@main.command()
@click.argument(
    "benchmark",
    type=click.STRING,
)
@click.option(
    "--output-dir",
    help="Overrides the benchmark's output directory.",
    type=click.Path(
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        path_type=Path,
    ),
)
def v4_to_v5(benchmark: str, output_dir: Optional[Path]) -> None:
    """
    Converts output files from nmoo v4 to v5.

    Refer to the changelog for more details.
    """
    # pylint: disable=import-outside-toplevel
    import pandas as pd

    b = _get_benchmark(benchmark)
    _apply_overrides(b, output_dir=output_dir)
    for t in b.all_par_triples():
        path = (
            b._output_dir_path
            / f"{t.problem_name}.{t.algorithm_name}.{t.n_run}.pi.csv"
        )
        if not path.is_file():
            logging.warning("PI file %s not found", path)
            continue
        df = pd.read_csv(path)
        for pi in b._performance_indicators:
            col = "perf_" + pi
            if col not in df:
                logging.error("PI '%s' not present in %s", col, path)
                continue
            # pylint: disable=unsubscriptable-object
            tmp = df[[col, "algorithm", "problem", "n_gen", "n_run"]]
            tmp.to_csv(b._output_dir_path / t.pi_filename(pi))
        path.unlink()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
