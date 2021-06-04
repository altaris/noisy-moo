noisy-moo
=========

<img src="https://github.com/altaris/noisy-moo/raw/main/imgs/the_cow.png"
alt="The Cow" width="500"/>

An abstract pipeline for noisy multi-objective optimization, building upon [the
works](https://github.com/ERATOMMSD/QUATIC2021-KNN-Averaging) of
[Klikovits](https://klikovits.net) and
[Arcaini](http://group-mmm.org/~arcaini/).

# Getting started

See [example.py](/example.py) for a quick example.

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
typecheck it using [mypy](http://mypy-lang.org/).
