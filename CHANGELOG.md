Changelog
=========

# v5.0.0

## New features

* In simple use cases, gaussian noises can be specified more easily:
  ```py
  # Assume that the F component is numerical and 2-dimensional.

  # Before (still possible)
  mean = np.array([0., 0.])
  cov = .1 * np.eye(2)
  noisy_problem = nmoo.GaussianNoise(problem, parameters={"F": (mean, cov)})

  # Now
  noisy_problem = nmoo.GaussianNoise(problem, mean, cov)

  # Since cov is constand diagonal, the following is also possible
  noisy_problem = nmoo.GaussianNoise(problem, mean, .1)
  ```
* Added uniform noise wrapper, see `nmoo.noises.UniformNoise`.

## Breaking changes

* Class `nmoo.benchmark.Pair` has been replaced by `nmoo.benchmark.PAPair`,
  representing a problem-algorithm pair, and `nmoo.benchmark.PARTriple`,
  representing a problem-algorithm-(run number) triple. Method
  `nmoo.benchmark.Benchmark._all_pairs` has been replaced by
  `nmoo.benchmark.Benchmark.all_pa_pairs` and
  `nmoo.benchmark.Benchmark.all_par_triples`.
* Performance indicator files `<problem_name>.<algorithm_name>.<n_run>.pi.csv`
  are now split into
  `<problem_name>.<algorithm_name>.<n_run>.pi-<pi_name>.csv`, one for each
  performance indicator.
* `GaussianNoise.__init__`: The old parameter dicts must bow be passed as a
  key-value argument:
  ```py
  # Assume that the F component is numerical and 2-dimensional.

  # Old way, NO LONGER WORKS
  mean = np.array([0., 0.])
  cov = .1 * np.eye(2)
  noisy_problem = nmoo.GaussianNoise(problem, {"F": (mean, cov)})

  # New way
  noisy_problem = nmoo.GaussianNoise(problem, parameters={"F": (mean, cov)})
  ```
* `EvaluationPenaltyEvaluator.__init__`: In the past, the only supported
  penalty type was `"times"` (meaning that the perceived number of evaluations
  was the actual number times a certain coefficient). Since this will not
  change in the forseeable future, the `penalty_type` argument has been
  removed.
  ```py
  # Old way, NO LONGER WORKS
  evaluator = EvaluationPenaltyEvaluator("times", 5)

  # New way
  evaluator = EvaluationPenaltyEvaluator(5)
  ```
  Aditionally, the name of the argument is now `multiplier` (instead of the old
  `coefficient`).
  ```py
  # Old keyval style, NO LONGER WORKS
  evaluator = EvaluationPenaltyEvaluator(penalty_type="times", coefficient=5)

  # New keyval style
  evaluator = EvaluationPenaltyEvaluator(multiplier=5)
  ```

# v4.0.0

## Breaking changes

* In the wrapped problem histories, the `x` field has been renamed to `X`. This
  implies that the `x` field in history dumps are now called `X` instead.
* In argorithms descriptions, the `save_history` option is now ignored.
* When constructing a benchmark, the default performance indicator list now
  consist of only `igd` (instead of `gd`, `gd+`, `igd` and `igd+` previously).

# v3.0.0

## Beaking changes

* In the algorithm specification dictionaries of `Benchmark.__init__`, key
  `minimize_kwargs` is no longer considered. Instead, various other keys have
  been added. Refer to the documentation.

# v2.0.0

## Breaking changes

* `GaussianNoise.__init__` now takes (a dict of) multivariate Gaussian noise
  parameters as arguments. Previously, it took (a dict of) tuples indicating
  the mean and standard deviation of a 1-dimensional Gaussian noise that would
  then be applied to all components independently. This old behaviour can be
  replicated by specifying a diagonal covariance matrix, e.g. the following are
  equivalent:
  ```py
  # Assume that the F component is numerical and 2-dimensional.

  # Old way, NO LONGER WORKS
  noisy_problem = nmoo.GaussianNoise(problem, {"F": (0., 1.)})

  # New way
  mean = np.array([0., 0.])
  cov = np.array([
      [1., 0.],
      [0., 1.],
  ])  # Or more concisely, np.eye(2)
  noisy_problem = nmoo.GaussianNoise(problem, {"F": (mean, cov)})
  ```