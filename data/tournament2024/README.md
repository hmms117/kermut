# PET Tournament 2024 Dataset Placeholder

This directory is reserved for the AlignBio-hosted Protein Engineering Tournament 2024 dataset. The upstream repository
(https://github.com/alignbio/protein-engineering-tournament-2024) currently requires authentication and could not be
cloned from the execution environment, so the `vendor/` submodule has not been initialized yet.

Once access is available, add the dataset via:

```bash
git submodule add --name dataset_pet2024 https://github.com/alignbio/protein-engineering-tournament-2024 data/tournament2024/vendor
```

Use `interim/` and `processed/` to stage cleaned and harmonized outputs without modifying the vendor repository.
