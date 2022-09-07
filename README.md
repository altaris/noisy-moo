noisy-moo
=========

![Python 3](https://img.shields.io/badge/python-3-blue?logo=python)
[![MIT License](https://img.shields.io/badge/license-MIT-yellow)](https://choosealicense.com/licenses/mit/)
[![Code style](https://img.shields.io/badge/style-black-black)](https://pypi.org/project/black)
[![Maintainability](https://api.codeclimate.com/v1/badges/aedd8e97a23534a85bc5/maintainability)](https://codeclimate.com/github/altaris/noisy-moo/maintainability)
[![Documentation](https://badgen.net/badge/documentation/here/blue)](https://altaris.github.io/noisy-moo/nmoo.html)


<center>
    <img src="https://github.com/altaris/noisy-moo/raw/main/imgs/the_cow.png"
    alt="The C O W" width="200"/>
</center>


A wrapper-based framework for [pymoo](https://pymoo.org/) problem modification
and algorithm benchmarking. Initially developed to test
*KNN-averaging*[^quatic21].

# Installation

Simply run
```sh
pip install nmoo
```

# Getting started

## In a notebook

See
[example.ipynb](https://github.com/altaris/noisy-moo/blob/main/example.ipynb)
for a quick example.
[![Launch Google Colab
notebook](https://img.shields.io/badge/launch-colab-blue?logo=googlecolab)](https://colab.research.google.com/github/altaris/noisy-moo/blob/main/example.ipynb)

## For larger benchmarks

For larger benchmarks, you may want to use nmoo's CLI. First, create a module,
say [`example.py`](https://github.com/altaris/noisy-moo/blob/main/example.py),
containing your benchmark factory (a function that returns your
[benchrmark](https://altaris.github.io/noisy-moo/nmoo/benchmark.html#Benchmark)),
say `make_benchmark()`. Then, run it using
```sh
python -m nmoo run --verbose 10 example:make_benchmark
```
Refer to
```sh
python -m nmoo --help
```
for more information.

# Main submodules and classes

* `nmoo.benchmark.Benchmark`: A `Benchmark` object represents... a benchmark
  🤔. At construction, you can specify problems and algorithms to run, how many
  times to run them, what performance indicators to compute, etc. Refer to
  `nmoo.benchmark.Benchmark.__init__` for more details.
* `nmoo.wrapped_problem.WrappedProblem`: The main idea of `nmoo` is to wrap
  problems in layers. Each layer should redefine `pymoo.Problem._evaluate` to
  intercept calls to the wrapped problem. It is then possible to apply/remove
  noise, keep a call history, log, etc.
* `nmoo.denoisers`: Sublasses of `nmoo.wrapped_problem.WrappedProblem` that
  implement denoising algorithms. In a simple scenario, a synthetic problem
  would be wrapped in a noise layer, and further wrapped in a denoising layer
  to test the performance of the latter.
* `nmoo.noises`: Sublasses of `nmoo.wrapped_problem.WrappedProblem` that apply
  noise.

# Contributing

## Dependencies

* `python3.8` or newer;
* `requirements.txt` for runtime dependencies;
* `requirements.dev.txt` for development dependencies (optional);
* `make` (optional).

Simply run
```sh
virtualenv venv -p python3.8
. ./venv/bin/activate
pip install -r requirements.txt
pip install -r requirements.dev.txt
```

## Documentation

Simply run
```sh
make docs
```
This will generate the HTML doc of the project, and the index file should be at
`docs/index.html`. To have it directly in your browser, run
```sh
make docs-browser
```

## Code quality

Don't forget to run
```sh
make
```
to format the code following [black](https://pypi.org/project/black/),
typecheck it using [mypy](http://mypy-lang.org/), and check it against coding
standards using [pylint](https://pylint.org/).




[^quatic21]: Klikovits, S., Arcaini, P. (2021). KNN-Averaging for Noisy
    Multi-objective Optimisation. In: Paiva, A.C.R., Cavalli, A.R., Ventura
    Martins, P., Pérez-Castillo, R. (eds) Quality of Information and
    Communications Technology. QUATIC 2021. Communications in Computer and
    Information Science, vol 1439. Springer, Cham.
    https://doi.org/10.1007/978-3-030-85347-1_36