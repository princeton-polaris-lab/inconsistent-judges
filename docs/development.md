# Developing the package

How to test a change, update a golden, and keep the repository readable. Read it before you send a change to the
code, the configs or the goldens.

## Tests

```bash
uv sync --group grpo                                  # the default groups and the training stack; no GPU needed
uv run pytest -n auto                                 # one worker per CPU this process may use
uv run pytest -n auto --cov --cov-report=json:coverage.json
uvx ruff@0.14.5 check src tests
```

| Note | |
|---|---|
| Markers | the default run deselects `gpu` (a CUDA device), `cluster` (the published data) and `slow` |
| Coverage | every file at 100% (`pyproject.toml`); without the `viz` and `grpo` groups the plotting and torch tests skip and it fails |
| Two coverage runs | each needs its own `COVERAGE_FILE` and report path |
| Frozen tests | `tests/test_*_frozen.py` hold each golden to its numbers |
| Release scan | `tests/test_release_scan.py` scans the checkout for secrets, private paths and addresses |
| Docs gate | `tests/test_docs_gate.py` parses every `ruleforge` command in the docs and checks that the module paths they name exist |
| Hydra programs | a test runs one through `run_program` from `tests/support/hydra_runs.py`, which clears Hydra's state and restores the process's logging |

## Update a golden

A golden changes only for a reason you can state, and no number moves by accident.

1. Rebuild twice: `ruleforge reproduce <experiment> rebuild --out DIR` with two different `DIR`s.
2. Check that the two builds are byte-equal.
3. Before you copy, run `ruleforge reproduce <experiment> verify --rerun DIR --exact` to see which golden differs, and
   `ruleforge reproduce <experiment> verify --rerun DIR` to see which numbers move. Give the reason in the commit
   message.
4. Copy the build's file over the golden under `results/`, then run the rebuild test below.

A change that reaches a golden's build is held to the rebuild of every golden from the roots it records:

```bash
RULEFORGE_REBUILD_DIR=<rebuild dir> RULEFORGE_ARTIFACT_ROOT=<artifact root> \
    uv run pytest -m cluster tests/identity/test_published_roots.py
```

It passes when every rebuilt golden equals the committed one as parsed JSON. Run it in a CPU job: it reads the published
roots' ledgers.

## Identity tests

Every published root is addressed by digests that are pure functions of bytes on disk. `tests/identity/` pins them on
recorded values, so a change that would move a published digest fails on CPU, before any golden is rebuilt.

| File | Pins |
|---|---|
| `test_identity_frozen.py` | every `condition.json` and `sampling_manifest.json` of the published roots, rewritten byte for byte; the regenerated ruleset-size subsets |
| `test_tables.py` | recorded generation seeds, request-block ids and prompt digests; recorded attempt rows classified again |
| `test_contracts.py` | the response contract of every built-in style and published condition; `style_families`; the prompt table |
| `test_constructors.py` | every published condition built again from its judge's and the task's identity; `-m cluster`: over its live weights |
| `test_digests.py` | `weights_digest`, `model_weights_identity`, `adapter_identity`; recorded paid-batch request sets |
| `test_fixture_check.py` | the fixture check itself, on a synthetic root |
| `test_published_roots.py` | `-m cluster`: the fixtures against the roots, and the rebuild above |
| `test_shipped_roots.py` | `published.ROOTS` is the set the Hub ships |

The fixtures hold what the records hold. They change only when the published set changes, so any other difference is a
finding.

```bash
RULEFORGE_ARTIFACT_ROOT=<artifact root> uv run python -m tests.identity.generate_fixtures           # a CPU job: reads GBs
RULEFORGE_ARTIFACT_ROOT=<artifact root> uv run python -m tests.identity.generate_fixtures --check   # rebuilds each row from its source line
RULEFORGE_ARTIFACT_ROOT=<artifact root> RULEFORGE_MODEL_ROOT=<model root> uv run python -m tests.identity.judge_descriptors
```

`judge_descriptors` also reads `RULEFORGE_WEIGHTS_DIRS`: derived checkpoints served from another machine's path.

## Pack the Hub repositories

`data/hub_inventory.yaml` lists what each Hub repository holds and pins every file; `data/hub_cards/` holds the cards.

```bash
uv run python -m ruleforge.hub pack --artifact-root <artifact root> --card data/hub_cards/dataset.md --out <new dir>
uv run python -m ruleforge.hub pack --model --artifact-root <artifact root> --card data/hub_cards/model.md --out <new dir>
```

Each writes a new directory, holds every file to its pin, and uploads nothing. The artifact root holds each campaign
root under `sampling/` and every other file where `ruleforge data fetch` places it. Every file
`ruleforge.hub.PUBLISHED_FIELDS` names is published with its local paths and host names rewritten by the table's rules.

## Recorded names

Strings the code must keep to read the published data live in `src/ruleforge/core/record_names.py`, and the prompt table
in `record_names.json`. Changing one breaks reading the data; [data/README.md](../data/README.md) holds the names note.

## Comments and docs

Written for a first-time adopter. These rules apply to source, configs, tests and docs. The code-style test that
enforces them is internal and omitted from the public tree.

| Rule | |
|---|---|
| No comment by default | comment only what the code cannot say: a non-obvious why, a unit or format, an outside constraint. One line, three at most; anything longer goes in `docs/` |
| No module docstrings | a runnable module passes argparse a one-line `description=` |
| Public functions and classes | one sentence saying what they return or do; private helpers none unless the logic is hard to follow |
| Present tense | no history, versions, plan or decision references, people, dates, job ids, cluster names or private paths; the same for error messages and `help=` |
| The paper's words | inconsistency rate, same-call and separate-call, leniency, ruleset size, prompt ablation, hierarchy, training; name the experiment, not a figure; "judgment", not "verdict", in reader-facing text |
| Configs | a header of at most three lines saying what the file runs; comment a key only for its unit or allowed values |
| Tests | the name states the behaviour; a docstring is optional and one line |
| Dead code | deleted, not documented |
| Docs pages | a page opens with what it is for, then its sections; a section starts with its command or smallest example; facts in tables, commands in code blocks, paragraphs at most three sentences; a term is defined where it first matters; figure numbers only in the table of [docs/reproduce.md](reproduce.md) |
