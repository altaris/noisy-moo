"""
noisy-moo CLI
"""
__docformat__ = "google"

import os
import re
import sys
from importlib import import_module
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from loguru import logger as logging

from nmoo.utils.logging import configure_logging


sys.path.append(os.getcwd())


def _apply_overrides(
    benchmark,
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
        logging.info("Overridden n_runs to {}", n_runs)
    if output_dir is not None:
        benchmark._output_dir_path = output_dir
        logging.info("Overridden output directory to {}", output_dir)
    if _include_exclude(benchmark._problems, only_problems, exclude_problems):
        logging.info(
            "Overridden problem list to {}",
            list(benchmark._problems.keys()),
        )
    if _include_exclude(
        benchmark._algorithms, only_algorithms, exclude_algorithms
    ):
        logging.info(
            "Overridden algorithm list to {}",
            list(benchmark._algorithms.keys()),
        )


def _get_joblib_kwargs(keyvals: List[str]):
    """
    Processes a list of the form `['key1=val1', ...]` into a dict that can be
    passed to `joblib.Parallel`.
    """
    ALL_JOBLIB_KEYS = [
        "n_jobs",
        "backend",
        "verbose",
        "timeout",
        "pre_dispatch",
        "batch_size",
        "temp_folder",
        "max_nbytes",
        "mmap_mode",
        "prefer",
        "require",
    ]
    kwargs: Dict[str, Any] = {}
    for keyval in keyvals:
        spl = keyval.split("=", maxsplit=2)
        if len(spl) != 2:
            logging.critical("Invalid joblib kwarg: '{}'", keyval)
            sys.exit(1)
        key: str = spl[0]
        val: Any = spl[1]
        if key not in ALL_JOBLIB_KEYS:
            logging.critical("Unknown joblib key: '{}'", key)
            sys.exit(1)
        if key == "n_jobs":
            logging.warning(
                "Overriding key 'n_jobs'. Use option '--n_jobs' or "
                "--n-post-processing-jobs' instead"
            )
        elif key == "verbose":
            logging.warning(
                "Overriding key 'verbose'. Use option '--verbose' instead"
            )
        if val.lower() in ["", "none"]:
            val = None
        else:
            try:
                val = float(val)
                if val == int(val):
                    val = int(val)
            except ValueError:
                pass
        kwargs[key] = val
    return kwargs


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


def _get_benchmark(path: str):
    """
    From a function "path" of the form `module[.submodule...]:function`,
    imports `module[.submodule...]` and returns `function`.
    """
    # pylint: disable=import-outside-toplevel
    try:
        import nmoo
    except ModuleNotFoundError:
        sys.path.append(str(Path(__file__).absolute().parent.parent))
        import nmoo
    try:
        module_name, function_name = path.split(":")
        module = import_module(module_name)
        factory = getattr(module, function_name)
        benchmark = factory()
        assert isinstance(benchmark, nmoo.Benchmark)
        return benchmark
    except AssertionError:
        logging.critical(
            "Factory '{}' did not return a Benchmark object.", path
        )
    except AttributeError:
        logging.critical(
            "Module '{}' has no attribute '{}'", module_name, function_name
        )
    except ModuleNotFoundError as e:
        logging.critical("{}", e)
    except TypeError:
        logging.critical("Factory '{}' is not callable.", function_name)
    sys.exit(-1)


@click.group()
@click.option(
    "-l",
    "--logging-level",
    default="INFO",
    help=(
        "Logging level, among 'debug', 'info', 'warning', 'error', and "
        "'critical'."
    ),
    type=click.STRING,
)
def main(logging_level: str) -> None:
    """noisy-moo CLI"""
    configure_logging(logging_level=logging_level)


@main.command()
@click.argument(
    "benchmark",
    type=click.STRING,
)
@click.option(
    "-ea",
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
    "-ep",
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
    "-oa",
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
    "-op",
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
    "-o",
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
def consolidate(
    benchmark: str,
    n_runs: Optional[int],
    only_problems: str,
    exclude_problems: str,
    only_algorithms: str,
    exclude_algorithms: str,
    output_dir: Optional[Path],
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
        output_dir=output_dir,
    )
    b.consolidate()


@main.command()
@click.argument(
    "benchmark",
    type=click.STRING,
)
@click.option(
    "-ea",
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
    "-ep",
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
    "--joblib-kwarg",
    help=(
        "A kwarg of the form 'key=value' to pass to joblib.Parallel.run (for "
        "both the benchmarking and post-processing phases)"
    ),
    multiple=True,
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
    "-oa",
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
    "-op",
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
    "-o",
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
    joblib_kwarg: List[str],
) -> None:
    """
    Runs a benchmark.

    Imports and executes a BENCHMARK, which is a string of the form
    'module[.submodule...]:function'. The 'function' returns the actual
    "Benchmark" object, and should be callable without any argument.
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
                **_get_joblib_kwargs(joblib_kwarg),
            )
        except KeyboardInterrupt:
            restart = False
        except Exception as e:  # pylint: disable=broad-except
            logging.error("Benchmark crashed: {}", e)
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
    "-ea",
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
    "-ep",
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
    "-oa",
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
    "-op",
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
    "-o",
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
    logging.info("Runs: {}/{}", n_run, len(all_triples))
    logging.info("GPPs: {}/{}", n_gpp, len(all_gpps))
    logging.info("PIs: {}/{}", n_pi, len(all_pis))


@main.command()
@click.argument(
    "benchmark",
    type=click.STRING,
)
@click.option(
    "-o",
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
            logging.warning("PI file {} not found", path)
            continue
        df = pd.read_csv(path)
        for pi in b._performance_indicators:
            col = "perf_" + pi
            if col not in df:
                logging.error("PI '{}' not present in {}", col, path)
                continue
            # pylint: disable=unsubscriptable-object
            tmp = df[[col, "algorithm", "problem", "n_gen", "n_run"]]
            tmp.to_csv(b._output_dir_path / t.pi_filename(pi))
        path.unlink()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
