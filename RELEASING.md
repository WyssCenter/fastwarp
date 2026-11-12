# Releasing fastwarp

## One-time setup

PyPI trusted publishing needs no API token. Register the project as a pending publisher before
the first release (PyPI: *Your account*, *Publishing*, *Add a new pending publisher*):

| Field | Value |
| --- | --- |
| PyPI project name | `fastwarp` |
| Owner | `WyssCenter` |
| Repository name | `fastwarp` |
| Workflow name | `wheels.yml` |
| Environment name | `pypi` |

In GitHub, create the environment `pypi` (*Settings*, *Environments*) before the first tag and
limit its deployment tags to `v*`; an environment that GitHub creates on first use has no
protection. Add a tag ruleset so that only maintainers can create `v*` tags. Add required
reviewers to the environment if a release should need a manual approval. The first successful
upload turns the pending publisher into a normal one.

## Release steps

1. Set `version` in `Cargo.toml` (`[workspace.package]`); the wheel takes its version from
   there. Update `version` in `CITATION.cff` to match.
2. In `CHANGELOG.md`, rename `## Unreleased` to `## <version> (<date>)`. The GitHub release
   notes are the section whose heading starts with the tagged version; the release fails if
   there is none. Add a new `## Unreleased` section with the next change.
3. Run the checks from [CONTRIBUTING.md](CONTRIBUTING.md) and merge to `main` with CI green.
4. Tag and push: `git tag v<version> && git push origin v<version>`.

## What the workflow does

`.github/workflows/wheels.yml` runs on the tag `v*`:

1. `build`: one abi3 wheel for each of Linux x86_64, Linux aarch64, macOS arm64, macOS x86_64 and
   Windows x86_64, with a smoke test and the Python tests where the runner allows.
2. `sdist`: `maturin sdist`.
3. `release`: fails unless the tag equals the `Cargo.toml` version, then creates the GitHub
   release with the wheels, the sdist and `SHA256SUMS`.
4. `publish`: after the three jobs above succeed, uploads the wheels and the sdist to PyPI
   through OIDC trusted publishing, in the `pypi` environment. Only this job has
   `id-token: write`.

A PyPI version cannot be replaced once uploaded. If a release is wrong, yank it on PyPI and
release the next version.

`gh workflow run wheels` builds the wheels by hand; the `publish` job runs only for a pushed
tag, never for a manual run.

## After the release

Check the project page at <https://pypi.org/project/fastwarp/> and install the wheel in a clean
environment: `python -c "import fastwarp; print(fastwarp.__version__)"`.
