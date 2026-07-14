# Contributing to AutoDataForge

Thanks for your interest in improving AutoDataForge! This guide covers how to set
up your environment, the quality bar we hold, and how to get changes merged.

## Ways to contribute

- **Report bugs** — open an issue with the *Bug report* template.
- **Request features** — open an issue with the *Feature request* template.
- **Improve docs** — typos, clarifications, and new guides are all welcome.
- **Submit code** — new agents, model adapters, export formats, or fixes.

## Development setup

```bash
git clone https://github.com/Prasadslaxmi08/autodataforge.git
cd autodataforge
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # add ,gui ,detect ,mcp as needed
```

## Architecture ground rules

AutoDataForge has a **frozen backend** and an **agent layer on top of it**. The most
important rule in this codebase:

> Agents orchestrate; they never reimplement backend logic.

Every agent drives the existing services through the `BackendController` seam (or the
`ToolRegistry`). If you find yourself duplicating pipeline, storage, or export logic
inside an agent, stop — wire a tool instead. See
[`docs/AGENT_ARCHITECTURE.md`](docs/AGENT_ARCHITECTURE.md).

## Quality bar (must pass before a PR)

```bash
ruff check .          # lint
mypy vds              # type checking (strict)
pytest -q             # tests
```

- **Tests are required** for new behavior. Match the existing style in `tests/`
  (no new frameworks or fixtures unless discussed).
- **Type hints** everywhere; the project runs `mypy --strict`.
- **No fabricated data** — if the backend can't supply a metric, surface it as
  *unavailable*; never invent numbers. This is a core project value.
- Keep diffs focused. One logical change per PR.

## Pull request process

1. Fork and create a branch: `git checkout -b feature/short-description`.
2. Make your change with tests and docs.
3. Ensure the full quality bar above passes locally.
4. Open a PR using the template; link the issue it closes.
5. A maintainer reviews; CI (Ruff + Pytest) must be green to merge.

## Commit messages

Use clear, imperative subjects (`Add VOC export writer`, `Fix report crash on
review-only plans`). Reference issues where relevant.

## Code of conduct

By participating you agree to uphold our
[Code of Conduct](CODE_OF_CONDUCT.md).
