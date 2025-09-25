#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${PROTEINMPNN_DIR:-}" ]]; then
    echo "PROTEINMPNN_DIR environment variable must be set to the ProteinMPNN checkout." >&2
    exit 1
fi

python_bin=${PYTHON_BIN:-python}

declare -a datasets
if [[ $# -gt 0 ]]; then
    datasets=("$@")
else
    assays_file=${ASSAYS_FILE:-data/DMS_substitutions.csv}
    if [[ ! -f "${assays_file}" ]]; then
        echo "Assay reference file not found: ${assays_file}" >&2
        exit 1
    fi
    mapfile -t datasets < <(awk -F"," 'NR > 1 {gsub(/"/, "", $3); if ($3 != "") print $3}' "${assays_file}")
fi

if [[ ${#datasets[@]} -eq 0 ]]; then
    echo "No assay identifiers supplied." >&2
    exit 1
fi

run_proteinmpnn() {
    local pdb_path=$1
    local dataset_id=$2

    if [[ ! -f "${pdb_path}" ]]; then
        echo "Skipping ${dataset_id}: missing PDB file at ${pdb_path}" >&2
        return
    fi

    local output_dir="data/conditional_probs/raw_ProteinMPNN_outputs/${dataset_id}/proteinmpnn"
    mkdir -p "${output_dir}"

    ${python_bin} "${PROTEINMPNN_DIR}/protein_mpnn_run.py" \
        --pdb_path "${pdb_path}" \
        --save_score 1 \
        --conditional_probs_only 1 \
        --num_seq_per_target 10 \
        --batch_size 1 \
        --out_folder "${output_dir}" \
        --seed 37
}

for dataset in "${datasets[@]}"; do
    echo "Processing ${dataset}..."
    if [[ "${dataset}" == "BRCA2_HUMAN" ]]; then
        for suffix in "_1-1000" "_1001-2085" "_2086-2832"; do
            run_proteinmpnn "data/structures/pdbs/${dataset}${suffix}.pdb" "${dataset}"
        done
    else
        run_proteinmpnn "data/structures/pdbs/${dataset}.pdb" "${dataset}"
    fi
    echo "Finished ${dataset}."
    echo
done
