#!/usr/bin/env bash

set -euo pipefail

dataset=${1:-TCRG1_MOUSE}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
"${script_dir}/conditional_probabilities.sh" "${dataset}"

echo "Generated conditional probabilities for ${dataset}."

if [[ "${dataset}" != "BRCA2_HUMAN" ]]; then
    echo "Note: BRCA2_HUMAN requires multiple PDB segments. See conditional_probabilities.sh." >&2
fi
