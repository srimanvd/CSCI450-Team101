# trustworthy-cli
A command-line interface for assessing the trustworthiness and quality of open-source AI models.

## Collaborators

* Sriman Donthireddi
* Duke Victor
* Tanay Patel

---

# Trustworthy CLI

A CLI tool to evaluate and score Hugging Face models, associated datasets, and codebases.

## Features

* **Simple CLI** with commands for installation, testing, and evaluation.
* **Comprehensive Metrics** including size, license compatibility, ramp-up time, bus factor, dataset/code availability, dataset quality, code quality, and performance claims.
* **Standardized Output** using NDJSON (Newline Delimited JSON) for easy integration with other tools.
* **Configurable Logging** to a file with adjustable verbosity levels for debugging.
* **High Code Quality** enforced with strong typing (`mypy`), style checks (`flake8`, `isort`), and test coverage.

---

## Usage

The project is managed through the `./run` script.

* `./run install` — Installs all necessary dependencies.
* `./run test` — Runs the test suite and prints a coverage report.
* `./run URL_FILE` — Evaluates models from a provided text file of URLs.

---

## Development

* **Language**: Python 3.9+
* **Key Libraries**: `huggingface_hub`, `PyGithub`, `gitpython`, `pytest`, `coverage`
* See the `./run` script for an overview of primary project commands.
