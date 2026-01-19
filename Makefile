.PHONY: check ruff-check ruff-format ruff-format-check pyright-check

pre-commit-install:
	pre-commit install

check:
	$(MAKE) ruff-format $(filter-out $@,$(MAKECMDGOALS)); \
	$(MAKE) ruff-format-check $(filter-out $@,$(MAKECMDGOALS)); \
	$(MAKE) ruff-check $(filter-out $@,$(MAKECMDGOALS)); \
	$(MAKE) pyright-check $(filter-out $@,$(MAKECMDGOALS)); \

# ruff
ruff-check:
	uv run ruff check $(filter-out $@,$(MAKECMDGOALS)) --config ./ruff.toml

ruff-format:
	uv run ruff format $(filter-out $@,$(MAKECMDGOALS)) --config ./ruff.toml

ruff-format-check:
	uv run ruff format --check $(filter-out $@,$(MAKECMDGOALS)) --config ./ruff.toml

# pyright
PYRIGHT_PYTHON_VERSION ?=

pyright-check:
	uv run pyright $(if $(PYRIGHT_PYTHON_VERSION),--pythonversion $(PYRIGHT_PYTHON_VERSION),) \
		$(filter-out $@,$(MAKECMDGOALS)) --project ./pyrightconfig.json
