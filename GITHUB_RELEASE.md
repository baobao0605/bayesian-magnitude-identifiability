# Manual GitHub publication instructions

This folder is ready to serve as a repository root after the author's final review and license decision. Repository creation and publication remain to be completed.

## Before upload

1. Review the manuscript, Supplement, code, result interpretation, and AI disclosures. Confirm that the title, author name, affiliation, email, and ORCID are correct.
2. Decide the license as described in `LICENSING.md`; add the actual license text if a license is chosen.
3. Keep the directory contents shown in `FILE_LIST.md`. Do not upload the surrounding `Submission/` or `Author_Review/` folders as part of this numerical repository. Do not add build caches, source-paper PDFs, private review records, credentials, or personal participant data.
4. If you want to check the supplied results before publishing, run `python code/reproduce.py`. The default does not add simulations or fits.

## Upload through the GitHub website

Create a new repository under your own account, choose its visibility, and upload the **contents** of this folder. The README should appear at the repository root alongside `code/`, `results/`, and `figures/`, rather than being nested inside another `GitHub_Repository/` directory. All supplied files are small enough for a conventional source repository; no dataset hosting service is required for this package.

For a first upload, use the prepared `CITATION.cff` as project metadata. It records the author and project title; add the repository URL and publication details when available.

## After a real repository URL exists

Copy the actual URL from the created repository. Add it to the README and manuscript's Data and Code Availability statement using the post-publication template provided in the separate author-review materials. Test the URL from a signed-out browser if it is intended to be public. For a public availability statement, the repository must be publicly accessible.

If you make a versioned release, describe it as the fixed materials accompanying the submitted manuscript. Keep the publication status current in the release description. An archival DOI can be added after the release is deposited with a service that issues one.

## What the default command changes

The default reproduction command refreshes deterministic check JSON files, the saved-result validation report, tail summaries, and five figure pairs. It writes new execution logs under `build/`, which is ignored by Git. The originally supplied recovery observations, fitted estimates, optimizer records, and interval tables are read without modification.

The commands containing `--rerun-simulation` or `--rerun-illustration` are optional routes for explicitly repeating the original fixed random procedures. They should be used in a copy if you wish to preserve all originally supplied numerical outputs.
