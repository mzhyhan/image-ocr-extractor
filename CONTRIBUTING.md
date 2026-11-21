# Contributing

Thanks for your interest in improving image-ocr-extractor!

## Development setup

```bash
sudo apt install tesseract-ocr        # or: brew install tesseract
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Checks

Every pull request must pass the same checks CI runs:

```bash
ruff check . && ruff format --check .   # lint + formatting
mypy                                    # strict type checking
pytest --cov                            # tests, doctests and coverage (>= 95%)
```

`pre-commit run --all-files` runs ruff, ruff-format, mypy and basic file hygiene hooks in one go.

## Guidelines

- Write Google-style docstrings for public modules, classes and functions.
- Add or update tests for any behaviour change; generate test images with Pillow
  instead of committing binary fixtures.
- Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, ...).
- Record user-facing changes in [CHANGELOG.md](CHANGELOG.md) under *Unreleased*.
