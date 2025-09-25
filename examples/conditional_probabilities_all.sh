#!/usr/bin/env bash

set -euo pipefail

assays_file=${ASSAYS_FILE:-data/DMS_substitutions.csv}
if [[ ! -f "${assays_file}" ]]; then
    echo "Assay reference file not found: ${assays_file}" >&2
    exit 1
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

awk -F"," 'NR > 1 {gsub(/"/, "", $3); if ($3 != "") print $3}' "${assays_file}" |
while IFS= read -r dataset; do
    "${script_dir}/conditional_probabilities.sh" "${dataset}"
done
