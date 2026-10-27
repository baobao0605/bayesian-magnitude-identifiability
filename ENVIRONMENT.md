# Numerical environment and execution scope

The retained simulation and current deterministic reproduction use these versions:

| Component | Version |
|---|---|
| Python | 3.12.14 |
| NumPy | 2.3.5 |
| SciPy | 1.17.0 |
| Matplotlib | 3.10.8 |
| SymPy | 1.14.0 |

The four Python packages are pinned in `requirements.txt`. Matplotlib uses its bundled DejaVu Sans font; no external font files are included. The public repository does not compile the article's LaTeX documents. The separate journal submission package contains those sources and build instructions.

## Randomness

- Recovery study: seed `2026102617`, with NumPy `SeedSequence.spawn` creating replicate streams and separate choice/reproduction/calibration streams. There are 200 paired replicates, one parameter point, and three optimizer starts per regime and replicate.
- Earlier numerical identity/likelihood-ridge illustration: seed `20261026`, with the original sequence of draws retained in `code/verify_equivalence.py`.
- Default `code/reproduce.py`: no random values are generated, no optimization is performed, and no simulation sample is added.

Both random-generation routes require an explicit command-line option. Neither was executed during the current submission-preparation round. That round instead reused the supplied synthetic arrays and fit records, evaluated likelihoods and intervals at the saved estimates, recomputed summaries, and redrew the five figures.

The actual current default run was launched from outside the repository directory and completed seven steps. Its standard output, standard error, commands, exit codes, and timings are supplied in `results/execution/`. The scripts resolve their own project paths. Installation and execution were checked in this Python 3.12 environment.

## Numerical comparisons

`code/validate_saved_results.py` uses relative tolerance `1e-8` and absolute tolerance `1e-9` for reconstructed intervals and nested summaries, with stricter checks for selected parameter and objective identities. Gradient and expected-information finite differences must have relative discrepancy below `1e-6`.

The software uses floating-point arithmetic and numerical optimization when the optional original simulation is explicitly rerun. Other numerical-library versions or platforms can produce small optimizer or figure-rendering differences. The manuscript's mathematical proofs do not depend on those software-specific numerical approximations.
