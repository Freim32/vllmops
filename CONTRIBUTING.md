# Contributing

Thanks for stopping by. The project is young and small, contributions of any size are welcome: bug reports, typo fixes, docs, new features, ideas. No formal proposal needed before a PR. If you are unsure, open a draft and we will figure it out together.

## Local setup

```bash
git clone https://github.com/Freim32/vllmops
cd vllmops
uv sync --extra dev
```

## Running checks

```bash
uv run poe checks
```

This runs `ruff format`, `ruff check`, `mypy`, `pyright` and `pytest`. CI runs the same on Python
3.10 through 3.14. For a coverage report, `uv run poe cov`.

## A few light conventions

- mypy strict and pyright standard, both over `src` and the test suite. `pyrightconfig.json` is what
  Pylance reads too, so the editor shows the same diagnostics as the terminal.
- Keep comments to the WHY, not the WHAT.
- Look at `git log` for the commit message shape.

That is it. Open an issue, a discussion, or a PR. Happy to chat.
