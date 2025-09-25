"""Utilities for preparing the QuaRALowP assay and running internal pipelines.

This module bundles two Typer commands:

``convert``
    Takes the raw Excel sheet exported from the QuaRALowP assay and converts it
    into the tab-separated layout consumed by the multi-objective tooling in
    this repository.

``run``
    Invokes ``convert`` and subsequently executes the Kermut-backed BOPO and
    BAGEL pipelines.  An optional command template can be supplied to launch an
    additional Kermut command on the freshly exported TSV.

Both commands try to stay flexible with respect to column naming because the
experiment sheet contains several concentration tabs and occasionally renamed
columns.  The defaults assume the commonly used column names in the assay: a
sequence column called ``sequence`` and objective columns called ``a450_raw``
and ``72C/25C_raw``.
"""

from __future__ import annotations

import shlex
import sys
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import typer


app = typer.Typer(help="Prepare QuaRALowP assay exports and run the selection pipelines.")


OBJECTIVE_COLUMNS = ("a450_raw", "72C/25C_raw")


def _load_excel_table(path: Path, sheet: Optional[str] = None) -> pd.DataFrame:
    """Return the requested sheet from ``path`` as a :class:`pandas.DataFrame`."""

    if not path.exists():
        raise FileNotFoundError(f"Input workbook not found at {path}")

    try:
        frame = pd.read_excel(path, sheet_name=sheet)
    except ValueError as exc:  # pragma: no cover - defensive handling of missing sheets
        raise ValueError(f"Unable to load sheet '{sheet}' from {path}") from exc

    # Normalise column names by stripping whitespace to avoid subtle mismatches.
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _detect_reference_sequence(df: pd.DataFrame, provided: Optional[str]) -> str:
    """Infer the reference sequence from ``df`` or fall back to ``provided``."""

    if provided:
        return provided

    candidates = [
        "reference_sequence",
        "Reference Sequence",
        "wt_sequence",
        "WT Sequence",
        "wild_type",
        "Wild Type",
    ]
    for column in candidates:
        if column in df.columns and df[column].notna().any():
            value = df[column].dropna().iloc[0]
            if isinstance(value, str) and value.strip():
                return value.strip()

    raise ValueError(
        "Reference sequence could not be inferred. Provide --reference-sequence explicitly."
    )


def _select_concentration(
    df: pd.DataFrame,
    *,
    column: Optional[str],
    value: Optional[str],
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Filter ``df`` to a specific concentration and return the chosen label."""

    if column is None or column not in df.columns:
        return df, value

    column_values = df[column].dropna().astype(str)
    if value is None or str(value).strip() == "":
        unique = sorted(column_values.unique())
        if len(unique) > 1:
            raise ValueError(
                "Multiple concentration values detected. Provide --concentration-value to choose one."
            )
        if unique:
            value = unique[0]
    mask = column_values == str(value)
    filtered = df.loc[mask].copy()
    if filtered.empty:
        raise ValueError(
            f"No rows matched concentration value '{value}'. Check --concentration-value."
        )
    return filtered, str(value)


def _prepare_columns(
    df: pd.DataFrame,
    *,
    sequence_column: str,
    objectives: Sequence[str],
    keep_columns: Iterable[str],
) -> pd.DataFrame:
    """Validate required columns and return a trimmed copy of ``df``."""

    if sequence_column not in df.columns:
        raise ValueError(
            f"Sequence column '{sequence_column}' missing from the worksheet."
        )

    missing_objectives = [col for col in objectives if col not in df.columns]
    if missing_objectives:
        raise ValueError(
            "Objective columns missing from the worksheet: " + ", ".join(missing_objectives)
        )

    trimmed = df.copy()
    trimmed = trimmed.rename(columns={sequence_column: "sequence"})
    trimmed = trimmed.dropna(subset=["sequence"])
    trimmed["sequence"] = trimmed["sequence"].astype(str).str.strip()
    trimmed = trimmed.loc[trimmed["sequence"].astype(bool)]

    # Keep requested metadata columns when present.
    columns_to_keep: List[str] = ["sequence"]
    columns_to_keep.extend(col for col in keep_columns if col in trimmed.columns)
    columns_to_keep.extend([col for col in objectives if col in trimmed.columns])
    columns_to_keep = list(dict.fromkeys(columns_to_keep))
    return trimmed[columns_to_keep]


def convert_assay(
    input_path: Path,
    output_path: Path,
    *,
    sheet: Optional[str],
    sequence_column: str,
    objectives: Sequence[str],
    concentration_column: Optional[str],
    concentration_value: Optional[str],
    keep_columns: Sequence[str],
    reference_sequence: Optional[str],
    batch: int,
    condition_label: Optional[str],
) -> str:
    """Convert the QuaRALowP Excel sheet into the agreed TSV format."""

    frame = _load_excel_table(input_path, sheet)
    frame, selected_concentration = _select_concentration(
        frame, column=concentration_column, value=concentration_value
    )

    prepared = _prepare_columns(
        frame,
        sequence_column=sequence_column,
        objectives=objectives,
        keep_columns=keep_columns,
    )

    reference = _detect_reference_sequence(frame, reference_sequence)

    prepared.insert(0, "batch", int(batch))
    condition = condition_label or selected_concentration or "batch_1"
    prepared.insert(1, "condition", str(condition))
    prepared.insert(2, "reference_sequence", reference)

    # Ensure objective columns are numeric when possible.
    for column in objectives:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared = prepared.dropna(subset=objectives, how="all")
    prepared = prepared.reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, sep="\t", index=False)

    return reference


def _format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _run_command(command: Sequence[str], *, working_dir: Optional[Path] = None) -> None:
    typer.secho(f"🚀 Running: {_format_command(command)}", fg=typer.colors.BLUE)
    subprocess.run(command, cwd=working_dir, check=True)


def _resolve_weights(
    values: Sequence[float],
    *,
    objectives: Sequence[str],
    fallback: float = 1.0,
    label: str,
) -> List[float]:
    """Return a weight per objective, expanding scalars and validating lengths."""

    if not objectives:
        raise ValueError("At least one objective column must be provided.")

    if not values:
        return [float(fallback)] * len(objectives)

    if len(values) == len(objectives):
        return [float(weight) for weight in values]

    if len(values) == 1:
        return [float(values[0])] * len(objectives)

    raise ValueError(
        f"{label} requires either one weight or as many weights as objectives (got {len(values)})."
    )


@app.command()
def convert(
    input_path: Path = typer.Argument(..., help="Path to the experiments/quaralowp XLSX file."),
    output_path: Path = typer.Argument(
        Path("experiments/quaralowp/quaralowp.tsv"),
        help="Destination TSV containing the formatted assay.",
    ),
    sheet: Optional[str] = typer.Option(None, help="Worksheet name inside the XLSX file."),
    sequence_column: str = typer.Option(
        "sequence", help="Column containing the amino-acid sequence."
    ),
    concentration_column: Optional[str] = typer.Option(
        None, help="Column describing assay concentration/condition."
    ),
    concentration_value: Optional[str] = typer.Option(
        None,
        help=(
            "Specific concentration to select. If omitted and multiple values are present,"
            " the command will raise an error."
        ),
    ),
    reference_sequence: Optional[str] = typer.Option(
        None, help="Override for the wild-type/reference sequence."
    ),
    batch: int = typer.Option(1, help="Batch identifier to stamp onto the export."),
    condition_label: Optional[str] = typer.Option(
        None,
        help="Optional string describing the condition. Defaults to the concentration value if available.",
    ),
    keep_column: List[str] = typer.Option(
        [],
        "--keep-column",
        help="Additional metadata columns to keep in the TSV when present.",
    ),
    objectives: List[str] = typer.Option(
        list(OBJECTIVE_COLUMNS),
        "--objective",
        help="Objective columns to retain (defaults to activity and 72C/25C).",
    ),
) -> None:
    """Convert the QuaRALowP XLSX export to the agreed TSV layout."""

    try:
        reference = convert_assay(
            input_path,
            output_path,
            sheet=sheet,
            sequence_column=sequence_column,
            objectives=objectives,
            concentration_column=concentration_column,
            concentration_value=concentration_value,
            keep_columns=tuple(keep_column),
            reference_sequence=reference_sequence,
            batch=batch,
            condition_label=condition_label,
        )
    except Exception as exc:  # pragma: no cover - surfaced to the CLI user
        raise typer.Exit(code=1) from exc

    typer.secho(
        f"✅ Wrote {output_path} with reference sequence length {len(reference)}",
        fg=typer.colors.GREEN,
    )


@app.command()
def run(
    input_path: Path = typer.Argument(..., help="Path to the experiments/quaralowp XLSX file."),
    output_dir: Path = typer.Option(
        Path("experiments/quaralowp"), help="Directory where artefacts will be stored."
    ),
    sheet: Optional[str] = typer.Option(None, help="Worksheet name to read from the Excel file."),
    sequence_column: str = typer.Option(
        "sequence", help="Column containing the amino-acid sequence."
    ),
    concentration_column: Optional[str] = typer.Option(
        None, help="Column describing assay concentration/condition."
    ),
    concentration_value: Optional[str] = typer.Option(
        None, help="Specific concentration entry to select from the worksheet."
    ),
    reference_sequence: Optional[str] = typer.Option(
        None, help="Override for the wild-type/reference sequence."
    ),
    condition_label: Optional[str] = typer.Option(
        None, help="Condition label stored in the exported TSV."
    ),
    keep_column: List[str] = typer.Option(
        [],
        "--keep-column",
        help="Additional metadata columns to keep in the TSV when present.",
    ),
    objectives: List[str] = typer.Option(
        list(OBJECTIVE_COLUMNS),
        "--objective",
        help="Objective columns to retain (defaults to activity and 72C/25C).",
    ),
    bagel_batch_size: int = typer.Option(24, help="Batch size for the BAGEL-style pipeline."),
    bopo_batch_size: int = typer.Option(24, help="Batch size for the BOPO-style pipeline."),
    bagel_weights: List[float] = typer.Option(
        [],
        "--bagel-weight",
        help="Objective weights passed to the BAGEL pipeline (defaults to 1 per objective).",
    ),
    bopo_weights: List[float] = typer.Option(
        [],
        "--bopo-weight",
        help="Objective weights for the BOPO pipeline (defaults to BAGEL weights).",
    ),
    bagel_seed: Optional[int] = typer.Option(None, help="Random seed for the BAGEL planner."),
    bopo_seed: Optional[int] = typer.Option(None, help="Random seed for the BOPO planner."),
    run_bagel: bool = typer.Option(True, help="Execute the BAGEL Monte Carlo pipeline."),
    run_bopo: bool = typer.Option(True, help="Execute the BOPO acquisition pipeline."),
    kermut_checkpoint: Path = typer.Option(
        ...,
        help="Path to the Kermut checkpoint used by the BAGEL and BOPO planners.",
    ),
    kermut_device: str = typer.Option(
        "cuda",
        help="Device passed to the Kermut-backed planners (e.g. cuda or cpu).",
    ),
    kermut_command: Optional[str] = typer.Option(
        None,
        help=(
            "Optional shell command template to run the Kermut one-liner."
            " Use {assay}, {wild_type}, and {output} placeholders to reference the"
            " generated TSV, the inferred wild-type sequence, and the desired output directory."
        ),
    ),
    kermut_output_dir: Path = typer.Option(
        Path("experiments/quaralowp/kermut"),
        help="Output directory supplied to the Kermut command template.",
    ),
) -> None:
    """Convert the assay and run the BOPO/BAGEL planners (plus optional Kermut)."""

    output_dir.mkdir(parents=True, exist_ok=True)
    assay_path = output_dir / "quaralowp.tsv"
    try:
        reference = convert_assay(
            input_path,
            assay_path,
            sheet=sheet,
            sequence_column=sequence_column,
            objectives=objectives,
            concentration_column=concentration_column,
            concentration_value=concentration_value,
            keep_columns=tuple(keep_column),
            reference_sequence=reference_sequence,
            batch=1,
            condition_label=condition_label,
        )
    except Exception as exc:  # pragma: no cover - surfaced to the CLI user
        raise typer.Exit(code=1) from exc

    objectives = list(objectives)
    directions = ["max" if col == OBJECTIVE_COLUMNS[0] else "min" for col in objectives]
    bagel_weights_resolved = _resolve_weights(
        bagel_weights,
        objectives=objectives,
        fallback=1.0,
        label="BAGEL weights",
    )
    if bopo_weights:
        bopo_weights_resolved = _resolve_weights(
            bopo_weights,
            objectives=objectives,
            fallback=1.0,
            label="BOPO weights",
        )
    else:
        bopo_weights_resolved = bagel_weights_resolved

    if run_bagel:
        bagel_output = output_dir / "bagel_suggestions.csv"
        command: List[str] = [
            sys.executable,
            "-m",
            "helper_scripts.kermut_bagel",
            "suggest",
            "--data",
            str(assay_path),
            "--ckpt",
            str(kermut_checkpoint),
        ]
        for objective in objectives:
            command.extend(["--objectives", objective])
        for direction in directions:
            command.extend(["--directions", direction])
        for weight in bagel_weights_resolved:
            command.extend(["--weights", str(weight)])
        command.extend(["--batch-size", str(bagel_batch_size)])
        command.extend(["--out-csv", str(bagel_output)])
        command.extend(["--base-seq", reference])
        command.extend(["--device", kermut_device])
        if bagel_seed is not None:
            command.extend(["--seed", str(bagel_seed)])
        _run_command(command)

    if run_bopo:
        bopo_output = output_dir / "bopo_suggestions.csv"
        command = [
            sys.executable,
            "-m",
            "helper_scripts.kermut_bopo",
            "suggest",
            "--data",
            str(assay_path),
            "--ckpt",
            str(kermut_checkpoint),
        ]
        for objective in objectives:
            command.extend(["--objectives", objective])
        for direction in directions:
            command.extend(["--directions", direction])
        for weight in bopo_weights_resolved:
            command.extend(["--weights", str(weight)])
        command.extend(["--batch-size", str(bopo_batch_size)])
        command.extend(["--out-csv", str(bopo_output)])
        command.extend(["--base-seq", reference])
        command.extend(["--device", kermut_device])
        if bopo_seed is not None:
            command.extend(["--seed", str(bopo_seed)])
        _run_command(command)

    if kermut_command:
        format_kwargs = {
            "assay": str(assay_path),
            "wild_type": reference,
            "output": str(kermut_output_dir),
        }
        formatted = kermut_command.format(**format_kwargs)
        command = shlex.split(formatted)
        _run_command(command)

    typer.secho(
        f"🎉 Workflow completed. Assay TSV available at {assay_path}",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":  # pragma: no cover - Typer entry point
    app()

