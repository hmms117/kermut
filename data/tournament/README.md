# PET Pilot 2023 Dataset

This directory wraps the [Protein Engineering Tournament Pilot 2023](https://github.com/the-protein-engineering-tournament/pet-pilot-2023) dataset as a Git submodule.

- `vendor/` points to commit 666b1a4f711ca5ce278808b188bef9da0aa74b28.
- `interim/` is reserved for lightly cleaned tables derived from the vendor data.
- `processed/` is reserved for harmonized TSV/FASTA outputs ready for modeling workflows.

Downstream cleaning scripts should avoid modifying `vendor/` directly. Instead, read from the submodule and export standardized artifacts into `interim/` and `processed/`.
