# Contributing to LLManim

Thanks for your interest in contributing! This guide covers everything you need to get started.

## Development Setup

```bash
git clone https://github.com/rishabhbhartiya/LLManim.git
cd LLManim

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install Manim system dependencies (LaTeX, Cairo, etc.)
# See https://docs.manim.community/en/stable/installation.html
```

## Running Tests

```bash
pytest
pytest --cov=llmanim  # with coverage
```

## Code Style

This project uses `black` for formatting and `ruff` for linting.

```bash
black llmanim/
ruff check llmanim/
```

## Submitting a Pull Request

1. Fork the repo and create a branch: `git checkout -b feat/my-feature`
2. Make your changes with clear, focused commits
3. Add or update tests for your changes
4. Run `black` and `ruff` before committing
5. Open a PR against `main` and fill in the PR template

## Adding a New Component

- Place it in the appropriate subpackage (`attention/`, `base/`, `tokenization/`, etc.)
- Export it from the subpackage's `__init__.py`
- Add a usage example in the docstring
- Update `README.md` tables if it's a new public class or function

## Reporting Bugs

Use the [bug report template](https://github.com/rishabhbhartiya/LLManim/issues/new?template=bug_report.md).
Please include your Manim version, Python version, and a minimal reproduction script.

## Questions

Open a [GitHub Discussion](https://github.com/rishabhbhartiya/LLManim/discussions) for questions,
ideas, or show-and-tell of videos you've made with LLManim.