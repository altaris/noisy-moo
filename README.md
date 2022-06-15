noisy-moo
=========

![Python 3](https://img.shields.io/badge/python-3-blue?logo=python)
[![MIT License](https://badgen.net/badge/license/MIT/yellow)](https://choosealicense.com/licenses/mit/)
[![Code style](https://badgen.net/badge/style/black/black)](https://pypi.org/project/black)
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
pip install git+https://github.com/altaris/noisy-moo.git
```

## Getting started

### In a notebook

See
[example.ipynb](https://github.com/altaris/noisy-moo/blob/main/example.ipynb)
for a quick example.

### For larger benchmarks

For larger benchmarks, you may want to use nmoo's CLI. First, create a module,
say [`example.py`](https://github.com/altaris/noisy-moo/blob/main/example.py),
containing your benchmark factory (a function that returns your
[benchrmark](https://altaris.github.io/noisy-moo/nmoo/benchmark.html#Benchmark)),
say `make_benchmark()`. Then, run it using
```sh
python -m nmoo run --verbose 10 example:make_benchmark
```

Refer to `python -m nmoo --help` for more information.

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