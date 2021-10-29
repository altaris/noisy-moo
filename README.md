noisy-moo
=========

![Python 3](https://badgen.net/badge/Python/3/blue)
[![MIT License](https://badgen.net/badge/license/MIT/blue)](https://choosealicense.com/licenses/mit/)
[![Code style](https://badgen.net/badge/style/black/black)](https://pypi.org/project/black)
[![Maintainability](https://api.codeclimate.com/v1/badges/aedd8e97a23534a85bc5/maintainability)](https://codeclimate.com/github/altaris/noisy-moo/maintainability)

<img src="https://github.com/altaris/noisy-moo/raw/main/imgs/the_cow.png"
alt="The Cow" width="250"/>

A wrapper-based framework for pymoo problem modification. Motivated by [the
works](https://github.com/ERATOMMSD/QUATIC2021-KNN-Averaging) of
[Klikovits](https://klikovits.net) and
[Arcaini](http://group-mmm.org/~arcaini/).

# Installation

Simply run
```sh
pip install git+https://github.com/altaris/noisy-moo.git
```

## Getting started

See [example.ipynb](/example.ipynb) for a quick example. For larger benchmarks,
you may want to use nmoo's CLI. First, create a module, say `foobar.py`,
containing your benchmark factory (a function that returns your benchrmark),
say `make_benchmark()`. Then, run it using
```sh
nmoo run foobar:make_benchmark
```

Refer to `nmoo --help` for more information.


# Contributing

## Dependencies

* `python3.8` or newer;
* `requirements.txt` for runtime dependencies;
* `requirements.dev.txt` for development dependencies.
* `make` (optional);

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
