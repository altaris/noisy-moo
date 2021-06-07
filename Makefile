DOCS_PATH 		= docs
SRC_PATH 		= nmoo
VENV			= ./venv

.ONESHELL:

all: format typecheck lint

.PHONY: docs
docs:
	pdoc --output-directory $(DOCS_PATH) $(SRC_PATH)

.PHONY: docs-browser
docs-browser:
	pdoc $(SRC_PATH)

.PHONY: format
format:
	black --line-length 79 --target-version py38 $(SRC_PATH)

.PHONY: lint
lint:
	pylint $(SRC_PATH)

.PHONY: typecheck
typecheck:
	mypy -p $(SRC_PATH)
