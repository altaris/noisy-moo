DOCS_PATH 		= docs
SRC_PATH 		= nmoo
VENV			= ./venv

.ONESHELL:

all: format typecheck lint

.PHONY: docs
docs:
	-mkdir $(DOCS_PATH)
	pdoc --output-directory $(DOCS_PATH) $(SRC_PATH)

.PHONY: docs-browser
docs-browser:
	pdoc --math $(SRC_PATH)

.PHONY: format
format:
	black --line-length 79 --target-version py38 $(SRC_PATH) setup.py example.py

.PHONY: lint
lint:
	pylint $(SRC_PATH) setup.py example.py

.PHONY: typecheck
typecheck:
	-mypy -p $(SRC_PATH)
	-mypy setup.py
	-mypy example.py
