# AAV Fit4Function Dataset

This directory encapsulates the [vector-engineering/fit4function](https://github.com/vector-engineering/fit4function) dataset as a Git submodule.

- `vendor/` is pinned to commit 6bfc2ebfe4abcd45cc6fe737e4700242a5090fee.
- `interim/` should host lightly cleaned tables with consistent column naming.
- `processed/` should host harmonized TSV/FASTA exports for modeling pipelines.

All transformations should read from `vendor/` and write into `interim/` or `processed/` without mutating the upstream repository.
