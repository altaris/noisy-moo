Changelog
=========

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