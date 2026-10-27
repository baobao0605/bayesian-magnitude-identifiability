# GitHub release instructions for the existing repository

Repository: https://github.com/baobao0605/bayesian-magnitude-identifiability

The repository has been created. Its scientific code, supplied synthetic datasets, and saved results
are already present in the downloaded snapshot. Do not upload any old project folder over the repo.
This file records steps still to be completed, not a claim that any release has already been published.

## Publication checklist

1. The author decides whether to adopt the suggested MIT (code) and CC BY 4.0 (author-owned
   synthetic results, figures, and documentation) terms. If accepted, add both actual license files
   and update the README and `LICENSING.md` together. Licenses grant reuse permissions: decide
   before publishing them.
2. Update `CITATION.cff` to include the actual repository URL. Do not invent a paper DOI or
   journal publication metadata; the manuscript is currently unpublished.
3. Check repository visibility is **Public** before claiming public code availability in the paper.
4. Run the current default validation command from the repository root:

       python code/reproduce.py

   It validates previously saved numerical results and redraws the figures without performing new
   simulated fits. Do not run `--rerun-simulation` for this metadata-only update.
5. Once the metadata edits are committed, use **Releases → Draft a new release**. Suggested tag:
   `v1.0.0` and suggested title: `Reproducibility materials for an unpublished manuscript`.
   Describe it as code and synthetic results for the accompanying theoretical manuscript, not as
   a published paper. No separate PDF paper, original Chen participant dataset, or article DOI is
   included. Publish only after checking the release tag targets the updated default branch.
6. Copy the real repository URL (and the release URL after publication) into the manuscript's
   Data and Code Availability section. Do not refer to a release until it exists.
7. Open the GitHub URL in a signed-out browser to confirm public access. Use **Code → Download ZIP**
   to check that `README.md`, `CITATION.cff`, `code/`, `results/`, and `figures/` are present.

The original prepared repository does not need new participant data, new model fitting, or new
mathematical research. Keep the original scripts, design JSON, synthetic arrays, and results intact.
