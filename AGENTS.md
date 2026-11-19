# AGENTS.md — FastWarp

Rules for coding agents in this repository. They add to [CONTRIBUTING.md](CONTRIBUTING.md), which
holds the setup, the checks and the pull-request rules.

## Command line

- Keep the `fastwarp` command line as simple as possible. Add an option only when a common task
  cannot be done without it.
- One way to do each thing: no aliases, no second option that overlaps an existing one.
- Advanced settings belong in the recipe file, the Python API or `fastwarp ants-registration`
  (which takes the `antsRegistration` syntax users already know), not in new options.
- Reuse the names and patterns of existing options. When unsure whether something needs an option,
  leave it out and ask.

## Documentation

- A new or changed feature changes the documentation in the same change: the pages in `docs/`
  (guide, `cli.md`, `api.md`, `limitations.md`), the docstrings and `--help` text, the README when
  it describes the feature, and an entry under `## Unreleased` in `CHANGELOG.md`.
- A change is not done until its documentation is. The docs build must stay at 0 warnings
  (`sphinx-build -W`).
