"""Pipeline to fit Kermut on QLP-style TSV exports and suggest mutations."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / 'kermut'
if 'kermut' not in sys.modules:
    spec = importlib.util.spec_from_file_location('kermut', PACKAGE_ROOT / '__init__.py')
    module = importlib.util.module_from_spec(spec)
    module.__path__ = [str(PACKAGE_ROOT)]
    sys.modules['kermut'] = module
    spec.loader.exec_module(module)


import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
import warnings

from kermut.cmdline.one_liner import (
    ESMEmbedder,
    Mutation,
    apply_mutations,
    build_candidate_library,
    build_config,
    make_gp_inputs,
    mutations_to_string,
    parse_mutation_code,
    predict_posterior,
    select_top_single_mutations,
    standardize_targets,
    train_gp,
)
from kermut.data import Tokenizer

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AA_ALPHABET)}

WT_PATTERN = re.compile(r"WT", re.IGNORECASE)
MUTATION_PATTERN = re.compile(r"([A-Z])(\d+)([A-Z])")


def _parse_target_columns(raw_cfg: Dict[str, object]) -> Tuple[str, ...]:
    if "target_columns" in raw_cfg and raw_cfg["target_columns"] is not None:
        raw_columns = raw_cfg["target_columns"]
        if isinstance(raw_columns, (list, tuple)):
            columns = [str(col) for col in raw_columns if str(col)]
        else:
            columns = [str(raw_columns)]
    else:
        columns = [str(raw_cfg.get("target_column", "A405_raw"))]
    normalized = []
    for column in columns:
        column = column.strip()
        if not column:
            continue
        normalized.append(column)
    if not normalized:
        raise ValueError("At least one target column must be specified in the configuration.")
    return tuple(dict.fromkeys(normalized))


def sanitize_target_name(target: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", target)
    return sanitized.strip("._") or "target"


@dataclass
class WildTypeConfig:
    fill_residue: str = "A"
    sequence_padding: int = 0
    overrides: Dict[str, str] | None = None


@dataclass
class ModelConfig:
    kernel_config: Path
    lr: float = 0.1
    n_steps: int = 150
    use_gpu: bool = False
    top_k_singles: int = 20
    candidate_batch: int = 12
    max_candidates: int = 250
    min_candidate_size: int = 2
    max_candidate_size: int = 2


@dataclass
class ProteinMPNNConfig:
    probabilities_dir: Optional[Path] = None
    on_diagonal: float = 0.9

    @property
    def off_diagonal(self) -> float:
        return (1.0 - self.on_diagonal) / (len(AA_ALPHABET) - 1)


@dataclass
class ESMConfig:
    model_path: Optional[Path] = None
    toks_per_batch: int = 8192
    use_gpu: bool = False


@dataclass
class PipelineConfig:
    data_path: Path
    output_dir: Path
    target_columns: Sequence[str]
    mutation_column: str
    group_column: str
    metadata_columns: Sequence[str]
    wild_type: WildTypeConfig
    model: ModelConfig
    proteinmpnn: ProteinMPNNConfig
    esm: ESMConfig


def load_config(path: Path) -> PipelineConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    wild_type_cfg = WildTypeConfig(
        fill_residue=str(raw.get("wild_type", {}).get("fill_residue", "A"))[:1] or "A",
        sequence_padding=int(raw.get("wild_type", {}).get("sequence_padding", 0)),
        overrides={
            key: str(value)
            for key, value in (raw.get("wild_type", {}).get("overrides", {}) or {}).items()
        }
        or None,
    )

    model_cfg = ModelConfig(
        kernel_config=Path(raw["model"]["kernel_config"]),
        lr=float(raw["model"].get("lr", 0.1)),
        n_steps=int(raw["model"].get("n_steps", 150)),
        use_gpu=bool(raw["model"].get("use_gpu", False)),
        top_k_singles=int(raw["model"].get("top_k_singles", 20)),
        candidate_batch=int(raw["model"].get("candidate_batch", 12)),
        max_candidates=int(raw["model"].get("max_candidates", 250)),
        min_candidate_size=int(raw["model"].get("min_candidate_size", 2)),
        max_candidate_size=int(raw["model"].get("max_candidate_size", 2)),
    )

    mpnn_cfg = ProteinMPNNConfig(
        probabilities_dir=(
            Path(raw["proteinmpnn"]["probabilities_dir"])
            if raw.get("proteinmpnn", {}).get("probabilities_dir")
            else None
        ),
        on_diagonal=float(raw.get("proteinmpnn", {}).get("on_diagonal", 0.9)),
    )

    esm_cfg = ESMConfig(
        model_path=(
            Path(raw["esm"]["model_path"])
            if raw.get("esm", {}).get("model_path")
            else None
        ),
        toks_per_batch=int(raw.get("esm", {}).get("toks_per_batch", 8192)),
        use_gpu=bool(raw.get("esm", {}).get("use_gpu", False)),
    )

    return PipelineConfig(
        data_path=Path(raw["data_path"]),
        output_dir=Path(raw["output_dir"]),
        target_columns=_parse_target_columns(raw),
        mutation_column=str(raw.get("mutation_column", "Mutations")),
        group_column=str(raw.get("group_column", "Construct")),
        metadata_columns=tuple(raw.get("metadata_columns", [])),
        wild_type=wild_type_cfg,
        model=model_cfg,
        proteinmpnn=mpnn_cfg,
        esm=esm_cfg,
    )


def parse_mutation_codes(raw: str) -> List[str]:
    if raw is None:
        return []
    raw = str(raw).strip()
    if not raw or WT_PATTERN.fullmatch(raw):
        return []

    parts = re.split(r"[;,\s]+", raw)
    codes: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        match = MUTATION_PATTERN.fullmatch(part)
        if not match:
            raise ValueError(f"Invalid mutation annotation: {raw}")
        codes.append(match.group(1).upper() + match.group(2) + match.group(3).upper())
    return codes


def infer_wild_type(
    df: pd.DataFrame,
    construct: str,
    cfg: PipelineConfig,
) -> str:
    overrides = cfg.wild_type.overrides or {}
    if construct in overrides:
        sequence = overrides[construct].strip().upper()
        if not sequence:
            raise ValueError(f"Override for {construct} is empty.")
        return sequence

    codes: Dict[int, str] = {}
    for raw in df[cfg.mutation_column].fillna(""):
        for code in parse_mutation_codes(raw):
            wt = code[0]
            pos = int(code[1:-1])
            existing = codes.get(pos, wt)
            if existing != wt:
                raise ValueError(
                    f"Conflicting wild-type residues at position {pos}: {existing} vs {wt}"
                )
            codes[pos] = wt

    if not codes:
        raise ValueError(f"Unable to infer wild-type for construct {construct}.")

    max_pos = max(codes) + int(cfg.wild_type.sequence_padding)
    sequence = [cfg.wild_type.fill_residue] * max_pos
    for pos, wt in codes.items():
        sequence[pos - 1] = wt
    return "".join(sequence)


def compute_frequency_embedding(sequence: str) -> np.ndarray:
    counts = np.zeros(len(AA_ALPHABET), dtype=np.float32)
    for aa in sequence:
        idx = AA_TO_INDEX.get(aa)
        if idx is not None:
            counts[idx] += 1.0
    if counts.sum() == 0:
        return counts
    return counts / counts.sum()


def pseudo_conditional_probabilities(
    wild_type: str, cfg: ProteinMPNNConfig
) -> np.ndarray:
    probs = np.full((len(wild_type), len(AA_ALPHABET)), cfg.off_diagonal, dtype=np.float32)
    for i, aa in enumerate(wild_type):
        idx = AA_TO_INDEX.get(aa)
        if idx is not None:
            probs[i, idx] = cfg.on_diagonal
        else:
            probs[i, :] = 1.0 / len(AA_ALPHABET)
    return probs


def aggregate_variants(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    construct: str,
    wild_type: str,
    tokenizer: Tokenizer,
    embedder: Optional[ESMEmbedder],
    target_column: str,
) -> pd.DataFrame:
    grouped = df.groupby(cfg.mutation_column, dropna=False)

    records: List[Dict[str, object]] = []
    for raw_mut, group in grouped:
        codes = parse_mutation_codes(raw_mut)
        mutations: List[Mutation] = [parse_mutation_code(code, wild_type) for code in codes]
        mutated_seq = apply_mutations(wild_type, mutations)

        target_values = pd.to_numeric(group[target_column], errors="coerce").dropna()
        if target_values.empty:
            continue

        record = {
            "mutations_raw": raw_mut if pd.notna(raw_mut) else "WT",
            "mutation_list": mutations,
            "mutations": mutations_to_string(mutations),
            "mutated_sequence": mutated_seq,
            "num_mutations": len(mutations),
            "fitness": float(target_values.mean()),
            "fitness_std": float(target_values.std(ddof=0)) if len(target_values) > 1 else 0.0,
            "n_replicates": int(len(target_values)),
        }
        # Preserve optional metadata columns when they have consistent values across replicates.
        for column in cfg.metadata_columns:
            if column in group:
                values = group[column].dropna().unique()
                record[column] = values[0] if len(values) == 1 else json.dumps(values.tolist())
        records.append(record)

    if not records:
        raise ValueError(f"No valid variants found for construct {construct}.")

    df_variants = pd.DataFrame.from_records(records)
    df_variants = df_variants.sort_values("mutations").reset_index(drop=True)

    sequences = df_variants["mutated_sequence"].tolist()
    if embedder is not None:
        embeddings = embedder.embed(sequences)
    else:
        embeddings = np.stack([compute_frequency_embedding(seq) for seq in sequences])
    df_variants["embedding_index"] = range(len(df_variants))

    tokens = tokenizer(sequences)
    df_variants.attrs["embeddings"] = embeddings
    df_variants.attrs["tokens"] = tokens
    return df_variants


def prepare_gp_inputs(
    df_variants: pd.DataFrame,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, np.ndarray]:
    embeddings = df_variants.attrs["embeddings"]
    emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
    tokens_attr = df_variants.attrs.get("tokens")
    if isinstance(tokens_attr, torch.Tensor):
        tokens_tensor: Optional[torch.Tensor] = tokens_attr.clone().detach()
    elif tokens_attr is not None:
        tokens_tensor = torch.tensor(tokens_attr, dtype=torch.long)
    else:
        tokens_tensor = None
    targets = torch.tensor(df_variants["fitness"].to_numpy(dtype=np.float32))
    return tokens_tensor, emb_tensor, embeddings


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    if "mutation_list" in df.columns:
        df = df.copy()
        df["mutation_list"] = df["mutation_list"].apply(
            lambda muts: ";".join(m.to_code() for m in muts) if muts else "WT"
        )
    df.to_csv(path, index=False)


def summarise_predictions(
    df: pd.DataFrame, *, num_single: int, num_double: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    singles = df[df["num_mutations"] == 1].copy()
    singles = singles.sort_values("posterior_mean", ascending=False).head(num_single)

    doubles = df[df["num_mutations"] == 2].copy()
    doubles = doubles.sort_values("posterior_mean", ascending=False).head(num_double)
    return singles, doubles


def run_pipeline(cfg: PipelineConfig) -> None:
    df = pd.read_csv(cfg.data_path, sep="\t")

    tokenizer = Tokenizer()
    embedder: Optional[ESMEmbedder] = None
    if cfg.esm.model_path is not None:
        try:
            embedder = ESMEmbedder(
                cfg.esm.model_path,
                toks_per_batch=cfg.esm.toks_per_batch,
                use_gpu=cfg.esm.use_gpu,
            )
        except FileNotFoundError as exc:
            warnings.warn(
                f"ESM weights not found ({exc}). Falling back to frequency embeddings.",
                RuntimeWarning,
            )

    multiple_targets = len(cfg.target_columns) > 1
    for target_column in cfg.target_columns:
        if target_column not in df.columns:
            warnings.warn(
                f"Target column '{target_column}' not found in dataset. Skipping.",
                RuntimeWarning,
            )
            continue

        df_target = df.copy()
        df_target[target_column] = pd.to_numeric(df_target[target_column], errors="coerce")
        df_target = df_target.dropna(subset=[target_column])
        if df_target.empty:
            warnings.warn(
                f"No valid measurements for target '{target_column}'. Skipping.",
                RuntimeWarning,
            )
            continue

        constructs = df_target[cfg.group_column].dropna().unique()
        target_root = cfg.output_dir
        if multiple_targets:
            target_root = cfg.output_dir / sanitize_target_name(target_column)
        ensure_dir(target_root)

        for construct in constructs:
            subset = df_target[df_target[cfg.group_column] == construct].copy()
            if subset.empty:
                continue

            construct_dir = target_root / construct
            ensure_dir(construct_dir)

            wild_type = infer_wild_type(subset, construct, cfg)
            (construct_dir / "wild_type.fasta").write_text(
                f">{construct}\n{wild_type}\n", encoding="utf-8"
            )

            df_variants = aggregate_variants(
                subset,
                cfg,
                construct,
                wild_type,
                tokenizer,
                embedder,
                target_column,
            )
            tokens_tensor, embeddings_tensor, embeddings_np = prepare_gp_inputs(df_variants)

            embedding_file = (
                "esm2_embeddings.npy" if embedder is not None else "frequency_embeddings.npy"
            )
            np.save(construct_dir / embedding_file, embeddings_np)

            cond_probs_source = "fallback"
            cond_probs = None
            if cfg.proteinmpnn.probabilities_dir is not None:
                candidate = cfg.proteinmpnn.probabilities_dir / f"{construct}.npy"
                if candidate.exists():
                    cond_probs = np.load(candidate)
                    cond_probs_source = "file"
            if cond_probs is None:
                cond_probs = pseudo_conditional_probabilities(wild_type, cfg.proteinmpnn)
            np.save(construct_dir / "proteinmpnn_conditional_probs.npy", cond_probs)

            train_inputs = make_gp_inputs(tokens_tensor, embeddings_tensor, None)

            targets_std, mean, std = standardize_targets(
                df_variants["fitness"].to_numpy()
            )
            targets_std_tensor = targets_std.clone()
            if cfg.model.use_gpu and torch.cuda.is_available():
                train_inputs = tuple(x.cuda() for x in train_inputs)
                targets_std_tensor = targets_std_tensor.cuda()

            model_cfg = build_config(
                kernel_config=cfg.model.kernel_config,
                use_gpu=cfg.model.use_gpu,
                lr=cfg.model.lr,
                n_steps=cfg.model.n_steps,
                embedding_dim=embeddings_tensor.shape[1],
                use_zero_shot=False,
                use_structure_kernel=False,
            )

            gp_inputs: Dict[str, torch.Tensor] = {}
            gp, likelihood = train_gp(
                train_inputs=train_inputs,
                targets=targets_std_tensor,
                cfg=model_cfg,
                gp_inputs=gp_inputs,
            )

            mu_train, sigma_train = predict_posterior(
                gp=gp,
                likelihood=likelihood,
                inputs=train_inputs,
                mean=mean,
                std=std,
            )

            df_variants["posterior_mean"] = mu_train
            df_variants["posterior_std"] = sigma_train
            df_variants["z_scores"] = (
                df_variants["posterior_mean"] - df_variants["fitness"].mean()
            ) / (df_variants["posterior_std"] + 1e-6)

            save_dataframe(df_variants, construct_dir / "training_posterior.csv")

            singles, doubles = summarise_predictions(
                df_variants,
                num_single=cfg.model.candidate_batch,
                num_double=cfg.model.candidate_batch,
            )
            save_dataframe(singles, construct_dir / "top_single_mutants.csv")
            save_dataframe(doubles, construct_dir / "top_double_mutants_existing.csv")

            candidates = None
            try:
                top_mutations = select_top_single_mutations(
                    df_variants, top_k=cfg.model.top_k_singles
                )
                candidates = build_candidate_library(
                    wild_type=wild_type,
                    top_mutations=top_mutations,
                    min_size=cfg.model.min_candidate_size,
                    max_size=cfg.model.max_candidate_size,
                    max_candidates=cfg.model.max_candidates,
                    existing_sequences=df_variants["mutated_sequence"].tolist(),
                )
            except ValueError:
                candidates = None

            if candidates is not None and not candidates.empty:
                candidate_sequences = candidates["sequence"].tolist()
                if embedder is not None:
                    candidate_embeddings = embedder.embed(candidate_sequences)
                else:
                    candidate_embeddings = np.stack(
                        [compute_frequency_embedding(seq) for seq in candidate_sequences]
                    )
                candidate_tensor = torch.tensor(candidate_embeddings, dtype=torch.float32)
                candidate_tokens = tokenizer(candidate_sequences)
                if cfg.model.use_gpu and torch.cuda.is_available():
                    candidate_tensor = candidate_tensor.cuda()
                    if isinstance(candidate_tokens, torch.Tensor):
                        candidate_tokens = candidate_tokens.cuda()
                candidate_inputs = make_gp_inputs(candidate_tokens, candidate_tensor, None)

                mu_cand, sigma_cand = predict_posterior(
                    gp=gp,
                    likelihood=likelihood,
                    inputs=candidate_inputs,
                    mean=mean,
                    std=std,
                )
                candidates["posterior_mean"] = mu_cand
                candidates["posterior_std"] = sigma_cand
                candidates = candidates.sort_values("posterior_mean", ascending=False).reset_index(
                    drop=True
                )
                candidates = candidates.head(cfg.model.candidate_batch)
                save_dataframe(candidates, construct_dir / "candidate_double_mutants.csv")

            summary = {
                "construct": construct,
                "target_column": target_column,
                "wild_type_length": len(wild_type),
                "num_variants": int(len(df_variants)),
                "num_single_mutants": int((df_variants["num_mutations"] == 1).sum()),
                "num_double_mutants": int((df_variants["num_mutations"] == 2).sum()),
                "generated_candidates": bool(candidates is not None and not candidates.empty),
                "used_esm_embeddings": embedder is not None,
                "proteinmpnn_source": cond_probs_source,
            }
            (construct_dir / "summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="YAML configuration file")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
