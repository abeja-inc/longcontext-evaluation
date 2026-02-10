.PHONY: check ruff-check ruff-format ruff-format-check pyright-check test-coverage

pre-commit-install:
	pre-commit install

check:
	$(MAKE) ruff-format $(filter-out $@,$(MAKECMDGOALS)); \
	$(MAKE) ruff-format-check $(filter-out $@,$(MAKECMDGOALS)); \
	$(MAKE) ruff-check $(filter-out $@,$(MAKECMDGOALS)); \
	$(MAKE) pyright-check $(filter-out $@,$(MAKECMDGOALS)); \

# ruff
ruff-check:
	uv run --frozen ruff check $(filter-out $@,$(MAKECMDGOALS)) --config ./ruff.toml

ruff-format:
	uv run --frozen ruff format $(filter-out $@,$(MAKECMDGOALS)) --config ./ruff.toml

ruff-format-check:
	uv run --frozen ruff format --check $(filter-out $@,$(MAKECMDGOALS)) --config ./ruff.toml

# pyright
PYRIGHT_PYTHON_VERSION ?=

pyright-check:
	uv run --frozen pyright $(if $(PYRIGHT_PYTHON_VERSION),--pythonversion $(PYRIGHT_PYTHON_VERSION),) \
		$(filter-out $@,$(MAKECMDGOALS)) --project ./pyrightconfig.json

pytest:
	uv run --frozen pytest -q tests
