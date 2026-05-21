CMD ?= init
ENV ?= dev
LICENSE=LICENSE
LICENSE_HEADER?=LICENSE_HEADER

.PHONY: venv install-deps clean-venv format

# Python environment management with uv
venv:
	uv venv .venv --python=python3.12
	.venv/bin/python -m ensurepip --upgrade
	@echo "Virtual environment created. Activate it with: source .venv/bin/activate"

install-deps:
	@if [ ! -f package-lock.json ]; then npm install; fi
	npm ci
	uv sync

clean-venv:
	rm -rf .venv

.PHONY: format
format: $(VENV)
	uv run black . --target-version=py312

develop:
	npm run offline

# Testing commands
.PHONY: test test-verbose test-streaming test-real-streaming test-agents test-quick test-specific test-summary test-lint test-format test-help test-three-agent test-models

PHONY: test
test: $(VENV)
	uv run coverage run \
		--omit="$(PWD)/tests" \
		-m pytest -vv $(args)

test-lint:
	uv run flake8 .

test-types:
	uv run mypy src
	uv run mypy tests

test-format:
	uv run black --check . --target-version=py312


.PHONY: cicd
cicd:
	cd infrastructure && make cicd CMD=$(CMD) ENV=$(ENV)


LICENSE:
	@echo "you must have a LICENSE file" 1>&2
	exit 1

LICENSE_HEADER:
	@echo "you must have a LICENSE_HEADER file" 1>&2
	exit 1

.PHONY: license
license: LICENSE LICENSE_HEADER $(VENV)
	uv run python -m licenseheaders -t ${LICENSE_HEADER} -d src $(args)
	uv run python -m licenseheaders -t ${LICENSE_HEADER} -d tests $(args)

.PHONY: test-license
test-license: LICENSE LICENSE_HEADER
	args="--check" $(MAKE) license
	
.PHONY: build-requirements
build-requirements: $(VENV)
	uv export --no-hashes --no-group dev > requirements-full.txt
	uv export --no-hashes --only-group base > requirements.txt

.PHONY: build-requirements-azure
build-requirements-azure: $(VENV)
	uv export --no-hashes --no-group dev > requirements.txt

.PHONY: test-all
test-all: test-lint test-types test-format test-license test