# Bayesian mechanisms in magnitude estimation: reproducibility materials

Companion code and synthetic results for **From observational equivalence to empirical commitments: Bayesian mechanisms in magnitude estimation**.

**Author:** Yongzhang Ji, Faculty of Psychology, Southwest University, Chongqing, China.  
**ORCID:** https://orcid.org/0009-0006-4111-2000  
**Contact:** royceji0605@email.swu.edu.cn

Prepared companion materials for an unpublished manuscript intended for the *Journal of Mathematical Psychology*.

## What is included

The materials support four analytical results: a rank-one restriction on weighting odds; a discrimination parameter ridge and a calibrated cross-task inverse; identification that depends on the response-noise reference; and bias-robust restrictions on logarithmic response means. Full proofs are in the accompanying manuscript and **Supplementary Material**, Notes S1–S7. This repository contains the numerical implementations and illustrations, not a substitute for those proofs.

All numerical observations are synthetic. There are no human participant records, third-party paper PDFs, or original-study participant data. The parameter-recovery example uses a single fixed parameter point with known additive reproduction noise. It does not reproduce the production-noise fit in Chen et al. (2024), perform model recovery, or validate a psychological mechanism.

| Location | Contents |
|---|---|
| `code/` | Mathematical checks, saved-result validation, the fixed simulation generator, and plotting scripts |
| `results/` | Fixed design, complete recovery arrays, fitted values, optimizer starts, intervals, summaries, and numerical-check outputs |
| `results/execution/` | Actual default-pipeline execution record and standard output/error files from submission preparation |
| `figures/` | All five manuscript figures, each in PDF and PNG |
| `requirements.txt` | Pinned numerical dependencies |
| `ENVIRONMENT.md` | Software versions, random seeds, and runtime details |
| `FILE_LIST.md` | Plain file inventory |
| `LICENSING.md` | License choices awaiting the author's decision |
| `GITHUB_RELEASE.md` | Manual upload and release instructions |

## Install

Use Python 3.12. The delivered calculations were executed with Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, Matplotlib 3.10.8, and SymPy 1.14.0. No TeX installation, system CJK font, browser, participant-data download, or earlier project directory is needed for these scripts.

From the repository root:

```bash
python -m venv .venv
```

Activate the environment with `. .venv/bin/activate` on macOS/Linux, or `.venv\Scripts\Activate.ps1` in Windows PowerShell, then run:

```bash
python -m pip install -r requirements.txt
python code/reproduce.py
```

Scripts resolve their inputs relative to their own location; the command may also be invoked with an absolute script path from another working directory.

## Default reproduction: use the supplied observations and fits

`python code/reproduce.py` performs seven steps:

1. Re-evaluate ten exact rational checks and the deterministic nonlinear examples.
2. Check eighteen symbolic identities.
3. Read the supplied synthetic arrays, 400 joint fits, 3,200 interval rows, and 1,200 optimizer-start rows. Re-evaluate the likelihood and interval calculation at the saved fitted points, and recompute the summaries.
4. Recompute the interval-width tail description and list the three computational-boundary fits.
5. Redraw Figures 1 and 2 directly from the stated equations.
6. Redraw Figures S1 and S2 from the existing ridge table and fixed equations.
7. Redraw Figure S3 from the supplied recovery estimates and summaries.

This default command performs **no random draws, no optimizer calls, and no new parameter fits**. It regenerates check outputs and figures. Per-step console records go to `build/run_<time>/`; `build/` is ignored by Git. Preserve an untouched copy of the supplied results if you want to compare a later run.

The submission-preparation run completed all seven steps. Its saved likelihood objectives agreed with re-evaluation from the supplied arrays to a maximum absolute difference of approximately `1.14e-13`. Reconstructed intervals and summaries agreed within the declared numerical tolerances. The actual record is in `results/execution/run_record.json`, and the scientific comparisons are in `results/saved_results_validation.json`.

## Optional complete regeneration of the original fixed recovery study

The original recovery study was already run before submission preparation. It contains **200 paired replicates** at `(w, v, w_m, g) = (.65, .5, .2, 1)`. Each replicate has 600 choice trials and 100 reproduction responses; the noisy-gain regime additionally uses 16 calibration observations. The complete design and numerical bounds are in `results/recovery_design.json`; the random seed is `2026102617`.

To explicitly regenerate and refit that same fixed study in a copy of the repository:

```bash
python code/reproduce.py --rerun-simulation
```

This command overwrites the recovery arrays, estimates, intervals, summaries, and figures. It does not enlarge the design. An equivalent standalone command is:

```bash
python code/recovery_simulation.py --rerun-simulation --overwrite
```

For a separate output folder, use `--output-dir` and `--figure-dir` on that standalone script. The supplied design remains the input. To redraw only the supplied recovery figure, use:

```bash
python code/recovery_simulation.py --plot-only
```

The full random generation and optimization path was **not run again during this submission-preparation round**. Its likelihood, gradient, information, optimizer, interval, and summary functions were retained from the existing executed study; the packaging edits add explicit execution guards and remove private release metadata.

## Earlier numerical illustration in Supplementary Note S8

`results/verification_results.json` records an earlier fixed-seed numerical check over 10,000 parameter draws and an illustrative 1,500-trial Bernoulli sample. Its seed is `20261026`. The 81-point likelihood-ridge table is supplied as `results/synthetic_ridge.csv`. The original random parameter arrays and individual choice observations were not retained; the seeded generator and saved ridge table are supplied.

To repeat the original seeded generation procedure:

```bash
python code/verify_equivalence.py --rerun-illustration
```

This optional command repeats the original seeded numerical checks and illustration, and replaces their check output, ridge table, and Figures S1/S2. It was not executed in this preparation round. The default workflow only redraws the already-supplied ridge table. The main proof checks and fixed-equation illustrations run without random sampling.

## Files and their relation to the paper

| Paper component | Code and supplied result |
|---|---|
| Four propositions and supporting examples | `code/rederive_core.py`, `code/check_claim_relations.py`; exact and symbolic result JSON files |
| Figure 1: empirical constraints | `code/make_revision_figures.py` → `figure1_empirical_constraints.*` |
| Figure 2: noise reference | `code/make_revision_figures.py` → `figure2_noise_reference.*` |
| Figure S1: discrimination ridge | `code/plot_saved_illustrations.py`, `results/synthetic_ridge.csv` → `figureS1_choice_ridge.*` |
| Figure S2: means versus likelihood | `code/plot_saved_illustrations.py` → `figureS2_mean_vs_likelihood.*` |
| Supplementary Note S9 and Figure S3 | `code/recovery_simulation.py`, the supplied recovery arrays/tables, and `code/recovery_tail_diagnostics.py` |

## Reading the recovery result

Speed RMSE was `0.14748` with known gain and `0.25504` with estimated gain. A valid 95% speed interval covered the truth in `190/200` and `187/200` replicates, respectively. Ignoring calibration uncertainty for the same noisy-gain point estimates reduced this count to `166/200`. The noisy-gain procedures have 197 valid intervals and three computational-boundary fits; those fits remain in point-estimate summaries and the all-replicate denominator.

Some interior log-Wald intervals are extremely wide. For propagated noisy-gain intervals, the median width is approximately `1.30834`, the 95th percentile is `7.95367`, and the maximum is approximately `1.87e6` degrees/s. The large tail is retained in `results/recovery_tail_diagnostics.json`. These are limitations of this particular finite-sample approximation, not exclusions or new experimental findings.

## Interpretation and reuse

Exact proofs, deterministic code checks, finite-sample recovery, and empirical mechanism validation are distinct. The numerical checks support implementation consistency; they do not establish a human mechanism. AI assisted mathematical verification, code development, literature work, and writing, as disclosed in the manuscript. Parallel AI checks are not independent human peer review.

`CITATION.cff` supplies the author's identity and project title. Add the repository URL and final article details when available. **Repository:** https://github.com/baobao0605/bayesian-magnitude-identifiability

## License

The software scripts under `code/` are released under the **MIT License** (see `LICENSE`).
The author-owned synthetic data, numerical results, figures, and documentation are released under
**Creative Commons Attribution 4.0 International (CC BY 4.0)** (see `LICENSE-DATA.md`).
The manuscript and Supplement are not part of this repository. These terms concern only
materials the author has the right to license; dependencies retain their respective licenses.

The accompanying paper is not yet published. `CITATION.cff` identifies this software repository;
a final article citation can be added after publication. A versioned GitHub Release can provide
a fixed code-and-data snapshot for the manuscript. See `GITHUB_RELEASE.md`.
