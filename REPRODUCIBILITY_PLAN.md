# EmbDI Paper Reproducibility Plan

**Status:** Active; Phases 0–4 complete (baseline, environment, data, runtime, determinism); Phase 5 in progress (synthetic smoke + one small dataset passed)
**Repository:** `/workspace`  
**Date:** 2026-10-23  
**Target:** Current `main`, CPython 3.12 only  
**Scope:** Paper-level reproduction of 8 datasets across ER, SM, and EQ (24 configs)  
**Out of scope:** Python 3.13/3.14 support, the `v1.0` legacy baseline, replacing Gensim, bit-identical paper results

## 1. Working Branch and Start Tag

Use the existing work branch and start tag:

- **Work branch:** `refactor-repro`
- **Start tag:** `repro-start` (lightweight tag)
- **Start commit:** `fec75703b7293ceaa907c6f481d1fd40ef406296`
- **Commit subject:** `📄 Updating license`
- **Commit date:** 2024-08-27

The original request referred to the branch as `refactor-repo`; the only matching branch present in the repository is `refactor-repro`, created from the latest commit. Use `refactor-repro` unless the branch is explicitly renamed.

Working rules:

- Stage this plan and all implementation changes on `refactor-repro`.
- Do not commit unless explicitly requested.
- Do not move, delete, or recreate `repro-start`.
- Keep `repro-start` at the commit above.
- Keep environment/data changes separate from behavioral code changes.
- Do not rewrite history or push from the sandbox.

Verification commands:

```bash
git switch refactor-repro
git show -s --format='%H %D%n%s' repro-start
git status --short --branch
```

## 2. Context and Decisions

The repository contains a `v1.0` tag labeled “Adding tag for final reproducibility version” at commit `8e9ef4db9cf9b339edc65a7ba878e31bf5c98835` from 2021-12-18. That tag pins `gensim==3.8.0`, which only supports CPython up to 3.7; `gensim 3.8.3` adds CPython 3.8. Faithful use of that tag therefore requires a legacy Python 3.7/3.8 environment.

The user chose to use **current `main` on Python 3.12** instead. Current `HEAD` is 41 commits beyond `v1.0`; it uses Gensim 4.3.3, renamed `EmbDI/` to `embdi/`, and includes refactors and bug fixes made after the paper.

Consequences:

- Python 3.12 is feasible because `gensim 4.3.3`, NumPy 1.26.4, pandas 2.2.2, scikit-learn 1.5.1, and networkx 3.3 all support it.
- This is not the tagged paper baseline. Results may differ from the paper because of post-paper code and dependency changes.
- Success means reproducing the pipeline and comparing against published results within the variance the README already documents, not producing bit-identical numbers.
- If differences are unexpectedly large, stop and reconsider the `v1.0` legacy baseline rather than silently changing the algorithm.

## 3. Current Known State

- The repository has 24 reproducibility configs: 8 entity resolution, 8 schema matching, 8 embeddings quality.
- All 24 configs validate after extraction of the Zenodo data; the extracted inputs remain excluded by `.gitignore`.
- Required data is excluded by `.gitignore`:
  - `pipeline/datasets`
  - `pipeline/edgelists`
  - `pipeline/test_dir`
  - `pipeline/embeddings`
  - `pipeline/walks`
- Tracked inputs already present include:
  - `pipeline/info/*.txt`
  - `pipeline/matches/er-matches/*.txt`
  - `pipeline/matches/sm-matches/*.txt`
- The data source is Zenodo record `7930461`, DOI `10.5281/zenodo.7930461`, concept DOI `10.5281/zenodo.7930460`, title “Datasets and configuration files for EmbDI: Embeddings for Data Integration.”
- The expected archive is `EmbDI datasets.zip`, size `1,421,696,261` bytes, MD5 `70036bc7482a6614d55d6054058ca17b`.
- The workspace mount is case-insensitive, so `EmbDI/` and `embdi/` resolve to the same directory in this sandbox. Other environments may not behave this way.
- There is no test suite.
- The project environment uses CPython 3.12.15 via `uv` (see Phase 1).
- `uv 0.13.4` is available and should be used for the environment.
- The README explicitly notes that ER and SM results can vary between runs because random walks and embedding training are stochastic.

Known issues identified at baseline (resolved in Phases 1–3):

- `pipeline/config_files/reproducibility/schema_matching/config-dblp_scholar-sm` points to a duplicated path:
  `pipeline/matches/sm-matches/sm-matches/sm-matches-dblp_scholar.txt`
- `setup.py` declares `packages=["EmbDI"]`, but the current package directory is `embdi`.
- `setup.py` performs directory creation at import time, which is undesirable for packaging.
- Several tasks write to fixed paths:
  - `pipeline/dump/indices.emb`
  - `pipeline/dump/sm_dump.emb`
  - `most_similar.pickle`
  - `suspicious_matches.csv`
  - `pipeline/generated-matches/...`
  This can overwrite artifacts between sequential configs.
- `embdi/schema_matching.py` contains `_clean_embeddings(embeddings_file)` at line 152, but the function signature is `_clean_embeddings(emb_file, matches)`. This may be in an unused path, but should be fixed.
- `main.py`’s config parser uses `key, value = line.split(':')`; it is fragile for values containing colons and does not handle inline comments.
- `generate_random_walks.py` has a malformed `parser.add_argument('')` call.
- `embdi/edgelist.py` has a tokenization argument bug and hardcoded `flatten=True`.
- `generate_config.py` has stale output directories and unconditional CSV reads.
- `data_preprocessing.py` imports `nltk`, `datasketch`, and `similarity`, and calls `nltk.download` at import time. It should not be imported unless regenerating edgelists.
- No global random seed is set. `random` and `numpy.random` are used in graph and walk generation; Gensim uses `workers=mp.cpu_count()` by default.
- Some graph/walk code iterates over sets, which can produce order-dependent behavior.

## 4. Reproduction Strategy

Freeze the current branch and commit at the start. Do not upgrade the pinned dependency set. Build a Python 3.12 environment, acquire the Zenodo data, validate all configs, fix only blocking/runtime issues, then run a staged verification ladder before launching all 24 configs.

The first pass should preserve current algorithm defaults so it remains comparable to the paper. Determinism controls should be optional and introduced only after or alongside the paper-like run.

## 5. Work Plan

### Phase 0 — Freeze the baseline

Tasks:

- Record the exact commit SHA, branch, and date.
- Record the Python 3.12 patch version.
- Confirm the 24-config scope.
- Record whether the paper table values are available separately; if not, note that only absolute metrics can be reported.

Deliverable:

- A short baseline record in the reproducibility document or commit message.

Gate:

- No work proceeds from an unrecorded/ambiguous commit.

#### Phase 0 baseline record (recorded 2026-10-23)

| Field | Value |
| --- | --- |
| Branch | `refactor-repro` |
| Start tag | `repro-start` (lightweight) |
| Commit SHA | `fec75703b7293ceaa907c6f481d1fd40ef406296` |
| Commit subject | `📄 Updating license` |
| Commit date | 2024-08-27T15:45:11+02:00 |
| Baseline recorded | 2026-10-23 |
| Target Python | CPython 3.12.15 (`cpython-3.12.15-linux-aarch64-gnu`, uv-managed, to be installed in Phase 1) |
| Current sandbox Python | CPython 3.14.8 (uv-managed); no system `python3` on `PATH` |
| Scope | 24 configs: 8 ER + 8 SM + 8 EQ |
| Platform | Linux arm64 |
| Paper table values | Not present in the repository; only absolute reproduced metrics can be reported unless sourced separately from paper DOI `10.1145/3318464.3389742` |

Scope confirmation (all files present under `pipeline/config_files/reproducibility/`):

- `entity_resolution/`: amazon_google, beer, dblp_acm, dblp_scholar, fodors_zagats, imdb_movielens, itunes_amazon, walmart_amazon
- `schema_matching/`: amazon_google, beer, dblp_acm, dblp_scholar, fodors_zagats, imdb_movielens, itunes_amazon, walmart_amazon
- `embeddings_quality/`: amazon_google, beer, dblp_acm, dblp_scholar, fodors_zagats, imdb_movielens, itunes_amazon, walmart_amazon

Notes:

- No separate paper result tables were found in the repository (`README.MD` only discusses run-to-run variance; no metric values).
- The working tree contains this plan as the only change; no behavioral code has been modified.

### Phase 1 — Python 3.12 environment

Tasks:

- Add `.python-version` containing `3.12`.
- Add `pyproject.toml` with:
  - `requires-python = ">=3.12,<3.13"`
  - the exact current pins:
    - `gensim==4.3.3`
    - `scikit-learn==1.5.1`
    - `numpy==1.26.4`
    - `pandas==2.2.2`
    - `tqdm==4.66.5`
    - `networkx==3.3`
- Do not upgrade dependencies.
- Generate `uv.lock` and record its hash.
- Fix `setup.py` packaging to use `embdi` and remove directory side effects, or make the project a non-package `uv` project if packaging is not needed.
- Keep `requirements.txt` for reference, but make the lockfile/environment canonical.
- Do not add preprocessing extras (`datasketch`, `nltk`, `similarity`) unless edgelist regeneration is required.

Suggested commands:

```bash
uv python pin 3.12
uv lock
uv sync
uv run python -c "import embdi, gensim, pandas, numpy, sklearn, networkx; print('ok')"
```

Deliverable:

- Reproducible Python 3.12 environment with a committed lockfile.

Gate:

- Imports succeed with no dependency builds from source.

#### Phase 1 environment record (2026-10-23)

| Field | Value |
| --- | --- |
| Interpreter | CPython 3.12.15 (`uv`-managed, `linux-aarch64-gnu`) |
| `uv` version | 0.13.4 |
| Lockfile | `uv.lock` (249 lines) |
| Lockfile SHA-256 | `bcea3d1514e44938e03db36865cf1182ca30306fcf866f3b803fc449ca37e2c3` |
| Resolved packages | 18 |
| Installed packages | 16 |
| Source builds | None (all wheels) |
| Packaging mode | Non-package uv project (`[tool.uv] package = false`) |

Files added/changed:

- `.python-version` → `3.12`; removed from `.gitignore` so it is tracked.
- `pyproject.toml` → `requires-python = ">=3.12,<3.13"` and the six exact pins.
- `setup.py` → `packages=["embdi"]`; import-time `os.makedirs` side effects removed.
- `requirements.txt` → unchanged, retained for reference.
- `uv.lock` → generated by `uv lock`; canonical environment source.

Key resolved versions: `gensim 4.3.3`, `scipy 1.13.1`, `numpy 1.26.4`, `pandas 2.2.2`, `scikit-learn 1.5.1`, `networkx 3.3`, `tqdm 4.66.5`, `smart-open 8.0.4`.

Verification:

```text
$ uv sync
Installed 16 packages in 4.06s   # no source builds

$ uv run python -c "import sys, embdi, gensim, pandas, numpy, sklearn, networkx, scipy, tqdm; ..."
ok 3.12.15 gensim 4.3.3 numpy 1.26.4 pandas 2.2.2 sklearn 1.5.1 networkx 3.3

$ uv run python -c "import embdi.embeddings, embdi.entity_resolution, ..."
core modules ok
```

Notes:

- Core modules import cleanly. Optional index backends `ngtpy`, `faiss`, and `annoy` are absent and emit `UserWarning` only; they are not required for the default path.
- No preprocessing extras (`datasketch`, `nltk`, `similarity`) were installed.

### Phase 2 — Acquire and validate data

Tasks:

- Fetch file links from the Zenodo record API for record `7930461`; do not hardcode guessed asset URLs.
- Download `EmbDI datasets.zip`.
- Verify size and MD5:
  - size `1,421,696,261`
  - MD5 `70036bc7482a6614d55d6054058ca17b`
- Inspect the archive layout before extracting.
- Extract datasets, edgelists, and EQ tests into the paths expected by the configs:
  - `pipeline/datasets/`
  - `pipeline/edgelists/`
  - `pipeline/test_dir/`
- Do not commit raw data. Add a downloader and checksum manifest instead.
- Validate all 24 configs after extraction.
- Fix the duplicated `sm-matches/sm-matches` path in `config-dblp_scholar-sm`.
- Preserve all other config semantics; do not “clean up” values that affect results.

Deliverable:

- Data present at expected paths, with checksums recorded.
- 24/24 configs pass path validation.

Gate:

- No config fails because of missing input.

#### Phase 2 data record (2026-10-23)

Zenodo metadata was read from the record API at run time (no asset URL was hardcoded). The record exposes exactly one archive matching the plan:

| Field | Value |
| --- | --- |
| Record / DOI | `7930461` / `10.5281/zenodo.7930461` |
| Concept DOI | `10.5281/zenodo.7930460` |
| Archive | `EmbDI datasets.zip` |
| Size | `1,421,696,261` bytes (verified) |
| MD5 | `70036bc7482a6614d55d6054058ca17b` (verified) |
| Local archive cache | `/tmp/embdi-repro/EmbDI datasets.zip` (not in git) |

Archive inspection showed the inputs under `EmbDI datasets/pipeline/`, plus extra `er datasets`, `er edgelists`, `sm datasets`, and `msd` trees that no reproducibility config references. Only these were extracted:

- `EmbDI datasets/pipeline/datasets/` → `pipeline/datasets/` (53 MB, 16 files)
- `EmbDI datasets/pipeline/edgelists/` → `pipeline/edgelists/` (342 MB, 24 files)
- `EmbDI datasets/pipeline/test_dir/` → `pipeline/test_dir/` (8.4 MB, 94 files)

Tracked `pipeline/info/` and `pipeline/matches/` were **not** overwritten. They were checked against the archive copies: 24 files identical, 0 different, 1 archive-only `er-matches/Readme.md`.

Added:

- `scripts/fetch_embdi_data.py` — derives the archive URL from the Zenodo API, downloads to `~/.cache/embdi-repro/`, verifies size and MD5, extracts only the three input trees, and verifies the extracted tree against the manifest. `--verify-only` checks without network.
- `scripts/embdi-data-manifest.json` — record/archive metadata plus SHA-256 and size for all 134 extracted input files (24 KB).

Config fix:

- `pipeline/config_files/reproducibility/schema_matching/config-dblp_scholar-sm`: `pipeline/matches/sm-matches/sm-matches/sm-matches-dblp_scholar.txt` → `pipeline/matches/sm-matches/sm-matches-dblp_scholar.txt`. This duplicated path typo is present in the upstream archive config as well.

Validation (using `main.read_configuration` + `embdi.utils.check_config_validity`):

```text
Before fix: 23/24 passed; only config-dblp_scholar-sm failed (missing match file)
After fix:  24/24 passed
Manifest:   Verified 134 files against manifest
```

The extracted data is ignored by git (`.gitignore` rules for `pipeline/datasets`, `pipeline/edgelists`, `pipeline/test_dir`); only the downloader and manifest are tracked.

### Phase 3 — Runtime and output-path fixes

Make minimal, behavior-preserving fixes. Do not change algorithm parameters.

Priority P0:

- Fix the SM config typo.
- Ensure per-run output isolation so sequential configs do not overwrite each other:
  - `pipeline/dump/indices.emb`
  - `pipeline/dump/sm_dump.emb`
  - `most_similar.pickle`
  - `suspicious_matches.csv`
  - `pipeline/generated-matches/*`
- Keep evaluation-facing filenames stable where the code expects them, but copy or namespace per run.
- Fix `_clean_embeddings` call signature in `embdi/schema_matching.py`.

Priority P1:

- Fix `main.py` config parsing for inline comments and values containing colons, only if encountered.
- Fix `generate_random_walks.py` if pre-generating walks is part of the run path.
- Fix `embdi/edgelist.py` only if regenerating edgelists from raw datasets.
- Do not use `--unblocking`; it hides failures.
- Avoid importing `data_preprocessing.py` unless raw-data preprocessing is explicitly in scope.

Deliverable:

- One real config can run end-to-end without clobbering unrelated artifacts.

Gate:

- Beer ER, SM, and EQ each complete successfully.

#### Phase 3 output-isolation record (2026-10-23)

P0 fixes made (behavior-preserving):

| Artifact | Before | After |
| --- | --- | --- |
| ER index dump | `pipeline/dump/indices.emb` | `pipeline/dump/indices-<emb-tag>.emb` |
| ER similarity dump | `most_similar.pickle` (repo root) | `pipeline/dump/most_similar-<emb-tag>.pickle` |
| ER suspicious matches | `suspicious_matches.csv` (repo root) | `pipeline/dump/suspicious_matches-<matches-tag>.csv` |
| SM filtered embeddings | `pipeline/dump/sm_dump.emb` | `pipeline/dump/sm_dump-<emb-tag>.emb` |

- Tags are derived from the input artifact basename (the embeddings or ground-truth file), which already embeds each config's `output_file` (`beer-ER`, `beer-SM`, `beer-EQ`, ...).
- `build_similarity_structure` now writes the pickle inside a context manager; `compare_ground_truth_only` now closes its CSV handle.
- `embdi/schema_matching.py::_clean_embeddings(emb_file, matches=None)` no longer requires the missing second argument, fixing the `match_columns` call at line 152. When `matches is None`, only `cid__` rows are kept. The ground-truth path used by `schema_matching` is unchanged.
- `pipeline/generated-matches/*`, `pipeline/embeddings/*`, and `pipeline/walks/*` were already namespaced by `output_file`/`run-tag`; no change needed.
- `.gitignore` now excludes generated run artifacts (`pipeline/dump`, `pipeline/generated-matches`, `pipeline/logging`, `pipeline/run-logs`, `pipeline/sim_files`, `pipeline/replacements`, `pipeline/experiments`, `pipeline/test`, `pipeline/*.params`). The tracked legacy `pipeline/dump/indices.emb` remains tracked but is left untouched.
- P1 items were not triggered: no config value contains a colon or inline comment; walk pre-generation and edgelist regeneration are not on the run path. `--unblocking` was not used and `data_preprocessing` was not imported.

Beer gate results (paper-like defaults, no seed, `workers=mp.cpu_count()`):

| Task | Metrics | Runtime |
| --- | --- | --- |
| ER | P 91.6667, R 80.8824, F 85.9375; 55 correct / 5 wrong / 68 matches | 356.88 s |
| SM | P 100.0000, R 100.0000, F 100.0000 | 388.04 s |
| EQ | MA 92.15, MR 78.29, MC 92.90, EQ_avg 87.78 | 193.19 s |

All three exited 0 with no traceback. Artifacts after all three runs:

```text
pipeline/dump/indices-beer-ER.emb
pipeline/dump/most_similar-indices-beer-ER.pickle
pipeline/dump/suspicious_matches-matches-beer.csv
pipeline/dump/sm_dump-beer-SM.emb
pipeline/embeddings/{beer-ER,beer-SM,beer-EQ}.emb
pipeline/walks/{beer-ER,beer-SM,beer-EQ}.walks
```

No root-level `most_similar.pickle` or `suspicious_matches.csv` was created, and no SM artifact clobbered an ER artifact. Logs were captured under `/tmp/embdi-repro/logs/` (ephemeral; metrics are recorded above). The pre-existing tracked `pipeline/dump/indices.emb` was left untouched.

### Phase 4 — Optional determinism controls

Tasks:

- Add `--seed` and `--workers` options without changing current defaults.
- When enabled:
  - seed Python `random`
  - seed NumPy
  - pass a seed to Gensim
  - set `workers=1`
  - set `PYTHONHASHSEED=0` before process start
- Sort set-derived iteration where order affects walks or graph construction.
- Keep a “paper-like” mode using existing defaults for the first comparison.
- Use seeded mode for repeated runs and variance estimates.

Deliverable:

- Optional reproducible mode documented separately from paper-like defaults.

Gate:

- Three seeded repeats of one small dataset produce stable outputs.

#### Phase 4 determinism record (2026-10-23)

Implemented an optional seeded mode without changing paper-like defaults:

- `main.py` adds `--seed` and `--workers`. With `--seed`, `full_run` seeds Python `random` and NumPy before each config, and `PYTHONHASHSEED=0` is enforced by re-executing the interpreter before the run starts.
- `embdi/embeddings.py::learn_embeddings` accepts `seed=None` and forwards it to `Word2Vec`, `FastText`, and `Doc2Vec`.
- `embdi/graph.py` accepts `deterministic=False`; seeded mode sorts the set-derived `to_link` contents by node name.
- `embdi/sentence_generation_strategies.py::generate_walks` sorts the intersection set in seeded mode.
- `--workers` defaults to `mp.cpu_count()` as before. When `--seed` is set and `--workers` is omitted, it defaults to `1`; an explicit `--workers` value is honored.
- Default runs (`no --seed`) take exactly the old paths: no RNG seeding, no sorting, `workers=mp.cpu_count()`.

Gate command (Fodors–Zagats ER, canonical config, seed 42, three repeats):

```bash
for i in 1 2 3; do
  uv run python main.py \
    -f pipeline/config_files/reproducibility/entity_resolution/config-fodors_zagats-er \
    --seed 42
done
```

All three repeats produced byte-identical artifacts:

| Artifact | SHA-256 |
| --- | --- |
| `fodors_zagats-ER.walks` | `70114ae3b391c0a47c44d59f9f1d51aa2b55c0e840e55b24dc98707538bb5c7b` |
| `fodors_zagats-ER.emb` | `728f458e95ee319db6029b9a3d1b11d44b138c47a7ccb5526837a20aef4099e7` |
| `indices-fodors_zagats-ER.emb` | `9252e7dc0569e88aeac22c425ebd1d3ad263cf2c8cd8a4988c7231caf5b48fa3` |
| `most_similar-indices-fodors_zagats-ER.pickle` | `a70f6f59c464f90e0a206ffce0d3b07f0968af8f10c32ca83210f794564042c0` |
| `suspicious_matches-matches-fodors_zagats.csv` | `67546e917e89c6eae3c25553176e2706195b8560011ceac6c016be0b8241c2f9` |

Repeated metrics were stable: P 99.0909, R 99.0909, F 99.0909; GP 64.4970, GR 99.0909, GF 78.1362. Runtimes were 159 s, 174 s, and 174 s (workers=1). A seeded Beer ER run also completed successfully (P 95.0820, R 85.2941, F 89.9225; 899 s), confirming the mode works on a larger config.

Seeded-mode metrics are not directly comparable to the paper-like Phase 3 numbers: sorting set-derived graph/walk order is a deliberate behavioral difference in seeded mode. Use paper-like defaults for the headline reproduction and seeded mode for repeatability/variance checks.

### Phase 5 — Verification ladder

Run in this order:

1. Import check and all-config validation.
2. Tiny synthetic edgelist:
   - graph generation
   - random walks
   - embeddings training
   - no full dataset
3. Small real dataset, preferably Beer:
   - ER
   - SM
   - EQ
4. Remaining datasets, one task family at a time:
   - ER
   - SM
   - EQ
5. Capture:
   - stdout/stderr
   - generated match files
   - embedding metadata
   - parameters and seed/workers
   - runtime

Do not launch all 24 configs until the synthetic smoke test and one real dataset pass.

Deliverable:

- Reproducible run log for one dataset, then the full matrix.

Gate:

- All 24 configs complete or every failure is explained.

#### Phase 5 smoke record (2026-10-23)

Synthetic smoke fixture (ephemeral, `/tmp/embdi-repro/synthetic/`): 20 cell nodes, 20 cid nodes, 60 value nodes, 80 edgelist rows, 200 walks of length 10, 20-dim Word2Vec, seed 7.

```bash
uv run python main.py -f /tmp/embdi-repro/synthetic/config-synthetic-smoke --seed 7
```

Result: exit 0; graph construction, random walks, and embedding training all completed in 0.02 s; 200/200 walks written; 41-line embedding file produced. A second seeded run produced byte-identical walks and embeddings (`cmp` on both). An unseeded default run also exited 0. No real dataset was read.

One real dataset (Beer ER/SM/EQ) already passed in Phase 3. After the Phase 4 changes, the full Fodors–Zagats ER/SM/EQ set passed again on paper-like defaults:

| Task | Metrics | Runtime |
| --- | --- | --- |
| ER | P 100.0000, R 100.0000, F 100.0000 | 67.72 s |
| SM | P 100.0000, R 100.0000, F 100.0000 | 58.83 s |
| EQ | MA 79.05, MR 88.43, MC 86.20, EQ_avg 84.56 | 40.51 s |

The remaining real configs have not been run.

### Phase 6 — Results and documentation

Tasks:

- Create a results table with:
  - dataset
  - task
  - paper value
  - reproduced value
  - delta
  - seed/workers
  - runtime
- Source paper values from the paper, DOI `10.1145/3318464.3389742`.
- Document known deviations:
  - current `main` is 41 commits past `v1.0`
  - Gensim 4.3.3 instead of 3.8.0
  - NumPy/pandas/scikit-learn/networkx version differences
  - post-paper refactors and bug fixes
  - documented run-to-run variance
- Add a reproducibility section to `README.MD` with exact commands.
- Record data checksum and lockfile hash.
- Keep raw data and generated artifacts out of git.

Deliverable:

- Forwardable reproducibility package: instructions, lockfile, checksums, results table, and limitations.

Gate:

- Another agent can repeat the smoke test from the documentation alone.

## 6. Acceptance Criteria

- Python 3.12 environment installs from the lockfile without source builds.
- 24/24 configs pass input validation after data placement.
- One small dataset runs ER, SM, and EQ end-to-end.
- Three seeded repeats produce stable results.
- A paper-versus-reproduced comparison table exists.
- All unresolved differences are documented.
- If paper deviations exceed normal documented variance, the team explicitly decides whether to fall back to `v1.0`.

## 7. Estimated Effort

- Environment and lockfile: 0.5 day
- Data acquisition and config validation: 0.5–1 day
- Runtime/output-path fixes: 1–2 days
- Determinism controls: 0.5–1 day
- Smoke and pilot dataset: 1 day plus compute
- Full 24-config runs and results write-up: compute-dependent plus 1 day

**Engineering estimate:** 3–5 days, excluding full experiment compute time.

## 8. Risks and Contingencies

- **Paper mismatch:** Current `main` is not the tagged paper baseline. If results diverge materially, stop and evaluate `v1.0` with Python 3.8/Gensim 3.8 rather than continuing to tune current code.
- **Silent output collisions:** Fixed dump paths can make later runs overwrite earlier artifacts. Fix before the full matrix.
- **Hidden failures:** Do not use `--unblocking`.
- **Data layout mismatch:** Verify archive contents before editing config paths.
- **Disk space:** The archive is about 1.4 GB compressed; extracted data and embeddings require additional space.
- **Architecture differences:** The sandbox is Linux arm64; record platform details because numerical results can vary across BLAS/CPU environments.
- **Nondeterminism:** Do not claim bit-identical reproduction. Report repeated runs and variance.

## 9. Out of Scope

- Python 3.13 and 3.14 support
- Running or porting the `v1.0` legacy tag
- Replacing Gensim or changing the embedding algorithm
- Regenerating raw datasets from original sources unless required
- Bit-identical reproduction of the paper’s tables

## 10. Handoff Checklist

- [x] Record commit SHA and Python 3.12 patch version.
- [x] Create and lock the Python 3.12 environment.
- [x] Download and checksum the Zenodo data.
- [x] Validate all 24 configs.
- [x] Fix the SM config path typo.
- [x] Fix output-path collisions.
- [x] Run synthetic smoke test.
- [x] Run Beer ER/SM/EQ.
- [x] Add optional seeded mode and verify three-repeat stability.
- [ ] Run full 24-config matrix.
- [ ] Produce paper-versus-reproduced results table.
- [ ] Document deviations and limitations.
- [ ] Keep raw data and generated artifacts out of git.
- [ ] Stage changes on `refactor-repro`; do not commit until requested.
