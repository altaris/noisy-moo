"""
noisy-moo CLI
"""
__docformat__ = "google"

import logging
import os
import re
import sys
from importlib import import_module
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
        module_name, factory_name = path.split(":")
        module = import_module(module_name)
        factory = getattr(module, factory_name)
        benchmark = factory()
        assert isinstance(benchmark, nmoo.benchmark.Benchmark)
        return benchmark
    except AssertionError:
        logging.fatal("Factory '%s' did not return a Benchmark object.", path)
    except AttributeError:
        logging.fatal(
            "Module '%s' has no attribute '%s'", module_name, function_name
        )
    except ModuleNotFoundError:
        logging.fatal("Module '%s' not found.", module_name)
    except TypeError:
        logging.fatal("Factory '%s' is not callable.", factory_name)
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
    b.run(
        n_jobs=n_jobs,
        n_post_processing_jobs=n_post_processing_jobs,
        verbose=verbose,
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
def tally(
    benchmark: str,
    n_runs: Optional[int],
    only_problems: str,
    exclude_problems: str,
    only_algorithms: str,
    exclude_algorithms: str,
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
    )
    all_pairs = b._all_pairs()
    all_gpps = {p.global_pareto_population_filename() for p in all_pairs}
    n_run = sum(
        map(
            lambda p: int(
                (b._output_dir_path / p.result_filename()).is_file()
            ),
            all_pairs,
        )
    )
    n_gpp = sum(
        map(lambda p: int((b._output_dir_path / p).is_file()), all_gpps)
    )
    n_pi = sum(
        map(
            lambda p: int((b._output_dir_path / p.pi_filename()).is_file()),
            all_pairs,
        )
    )
    logging.info("Runs: %d/%d", n_run, len(all_pairs))
    logging.info("GPPs: %d/%d", n_gpp, len(all_gpps))
    logging.info("PIs: %d/%d", n_pi, len(all_pairs))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
