"""Typer CLI to run embeddings, fit Kermut, and propose variants in one command."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import typer
from esm import FastaBatchedDataset, pretrained
from gpytorch.likelihoods import GaussianLikelihood
from omegaconf import OmegaConf
from kermut.data import Tokenizer
from kermut.gp import instantiate_gp, optimize_gp
from kermut.kernels import CompositeKernel


class Strategy(str, Enum):
    """Available exploration strategies."""

    MAX_DOE = "max_doe"
    ACTIVE_LEARNING = "active_learning"


DEFAULT_KERNEL_CONFIG = (
    Path(__file__).resolve().parent.parent / "hydra_configs" / "kernel" / "kermut_no_m_constant_mean.yaml"
)


@dataclass(frozen=True)
class Mutation:
    """A single amino-acid substitution described in ProteinGym convention."""

    position: int  # 1-indexed position in the sequence
    wild_type: str
    mutant: str

    def to_code(self) -> str:
        """Return the canonical mutation code (e.g., A23T)."""

        return f"{self.wild_type}{self.position}{self.mutant}"


class ESMEmbedder:
    """Wrapper around ESM-2 embedding extraction with caching."""

    def __init__(self, model_path: Path, toks_per_batch: int, use_gpu: bool) -> None:
        self.model_path = model_path
        if not self.model_path.exists():
            raise FileNotFoundError(f"ESM model not found at {self.model_path}")

        self.model, self.alphabet = pretrained.load_model_and_alphabet_local(
            str(self.model_path)
        )
        self.model.eval()

        device_type = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)
        if self.device.type == "cuda":
            self.model = self.model.to(self.device)

        self.toks_per_batch = toks_per_batch
        self._cache: Dict[str, np.ndarray] = {}

    def embed(self, sequences: Sequence[str]) -> np.ndarray:
        """Return mean-pooled embeddings for the requested sequences."""

        missing = [seq for seq in sequences if seq not in self._cache]
        if missing:
            dataset = FastaBatchedDataset(
                sequence_strs=missing,
                sequence_labels=missing,
            )
            batches = dataset.get_batch_indices(self.toks_per_batch, extra_toks_per_seq=1)
            data_loader = torch.utils.data.DataLoader(
                dataset,
                collate_fn=self.alphabet.get_batch_converter(truncation_seq_length=1022),
                batch_sampler=batches,
            )

            with torch.no_grad():
                for labels, seqs, toks in data_loader:
                    if self.device.type == "cuda":
                        toks = toks.to(device=self.device, non_blocking=True)
                    out = self.model(toks, repr_layers=[33], return_contacts=False)
                    reps = out["representations"][33].to(device="cpu")

                    for label, rep, seq in zip(labels, reps, seqs):
                        truncate = min(1022, len(seq))
                        embedding = rep[1 : truncate + 1].mean(dim=0).numpy()
                        self._cache[label] = embedding

        return np.stack([self._cache[seq] for seq in sequences])


def parse_mutation_code(code: str, wild_type: str) -> Mutation:
    """Parse a mutation code (e.g., A23T) into a :class:`Mutation`."""

    code = code.strip()
    if not code or code.upper() in {"WT", "-"}:
        raise ValueError("Wild-type mutation code does not describe a substitution.")

    if len(code) < 3:
        raise ValueError(f"Invalid mutation code: {code}")

    wt = code[0]
    mut = code[-1]
    try:
        pos = int(code[1:-1])
    except ValueError as exc:  # pragma: no cover - defensive programming
        raise ValueError(f"Invalid mutation code: {code}") from exc

    if wild_type[pos - 1] != wt:
        raise ValueError(
            f"Mutation {code} inconsistent with provided wild type at position {pos}."
        )

    return Mutation(position=pos, wild_type=wt, mutant=mut)


def apply_mutations(wild_type: str, mutations: Sequence[Mutation]) -> str:
    """Apply a list of mutations to the wild-type sequence."""

    seq = list(wild_type)
    for mut in mutations:
        seq[mut.position - 1] = mut.mutant
    return "".join(seq)


def derive_mutations_from_sequence(sequence: str, wild_type: str) -> List[Mutation]:
    """Return the list of substitutions required to reach ``sequence`` from wild type."""

    if len(sequence) != len(wild_type):
        raise ValueError("Sequences must be aligned to the wild type.")

    mutations: List[Mutation] = []
    for idx, (wt_aa, mut_aa) in enumerate(zip(wild_type, sequence), start=1):
        if wt_aa != mut_aa:
            mutations.append(Mutation(position=idx, wild_type=wt_aa, mutant=mut_aa))
    return mutations


def mutations_to_string(mutations: Sequence[Mutation]) -> str:
    """Create a canonical semicolon-separated mutation string."""

    if not mutations:
        return "WT"
    ordered = sorted(mutations, key=lambda m: m.position)
    return ";".join(m.to_code() for m in ordered)


def load_assay_table(
    path: Path,
    wild_type: str,
    sequence_col: str,
    target_col: str,
    mutation_col: str | None,
) -> pd.DataFrame:
    """Load TSV/CSV assay measurements and harmonise mutation annotations."""

    df = pd.read_csv(path, sep=None, engine="python")

    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in {path}.")

    if sequence_col not in df.columns and (mutation_col is None or mutation_col not in df.columns):
        raise ValueError(
            "Either a sequence column or a mutation annotation column must be provided."
        )

    if sequence_col not in df.columns:
        sequences = []
        mutation_lists: List[List[Mutation]] = []
        for raw_mut in df[mutation_col].fillna("WT"):
            raw_mut = str(raw_mut)
            if raw_mut.upper() in {"WT", "-", ""}:
                mutation_lists.append([])
                sequences.append(wild_type)
                continue

            codes = [code.strip() for code in raw_mut.replace(",", ";").split(";") if code.strip()]
            muts = [parse_mutation_code(code, wild_type) for code in codes]
            mutation_lists.append(muts)
            sequences.append(apply_mutations(wild_type, muts))

        df[sequence_col] = sequences
    else:
        mutation_lists = [
            derive_mutations_from_sequence(seq, wild_type)
            for seq in df[sequence_col].astype(str)
        ]

    df["mutation_list"] = mutation_lists
    df["mutations"] = df["mutation_list"].apply(mutations_to_string)
    df["sequence"] = df[sequence_col].astype(str)
    df["num_mutations"] = df["mutation_list"].apply(len)
    df["fitness"] = df[target_col].astype(float)

    if df["sequence"].str.len().nunique() != 1:
        raise ValueError("All sequences must share the same length as the wild type.")

    return df[["mutations", "mutation_list", "sequence", "num_mutations", "fitness"]]


def standardize_targets(values: Sequence[float]) -> Tuple[torch.Tensor, float, float]:
    """Return z-scored targets alongside mean and std for inverse transforms."""

    arr = np.asarray(values, dtype=np.float32)
    mean = float(arr.mean())
    std = float(arr.std() if arr.std() > 0 else 1.0)
    standardized = (arr - mean) / std
    return torch.tensor(standardized, dtype=torch.float32), mean, std


def build_config(
    kernel_config: Path,
    use_gpu: bool,
    lr: float,
    n_steps: int,
    embedding_dim: int,
    use_zero_shot: bool,
    use_structure_kernel: bool,
) -> OmegaConf:
    """Load and adapt the kernel configuration for the current run."""

    kernel_cfg = OmegaConf.load(kernel_config)
    kernel_cfg.embedding_dim = embedding_dim
    if "use_sequence_kernel" in kernel_cfg:
        kernel_cfg.use_sequence_kernel = True
    if "use_structure_kernel" in kernel_cfg:
        kernel_cfg.use_structure_kernel = use_structure_kernel
    if "use_zero_shot" in kernel_cfg:
        kernel_cfg.use_zero_shot = use_zero_shot

    cfg = OmegaConf.create(
        {
            "kernel": kernel_cfg,
            "use_gpu": use_gpu and torch.cuda.is_available(),
            "optim": {"lr": lr, "n_steps": n_steps, "progress_bar": False},
            "seed": 2024,
        }
    )
    return cfg


def train_gp(
    train_inputs: Tuple[torch.Tensor, ...],
    targets: torch.Tensor,
    cfg: OmegaConf,
    gp_inputs: Dict[str, torch.Tensor],
):
    """Fit the Kermut GP on the provided inputs."""

    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    gp, likelihood = instantiate_gp(
        cfg=cfg, train_inputs=train_inputs, train_targets=targets, gp_inputs=gp_inputs
    )
    gp, likelihood = optimize_gp(
        gp=gp,
        likelihood=likelihood,
        train_inputs=train_inputs,
        train_targets=targets,
        lr=float(cfg.optim.lr),
        n_steps=int(cfg.optim.n_steps),
        progress_bar=bool(cfg.optim.progress_bar),
    )

    return gp, likelihood


def predict_posterior(
    gp,
    likelihood: GaussianLikelihood,
    inputs: Tuple[torch.Tensor, ...],
    mean: float,
    std: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return posterior mean and std in the original fitness scale."""

    gp.eval()
    likelihood.eval()

    with torch.no_grad():
        posterior = likelihood(gp(*inputs))
        mu = posterior.mean.detach().cpu().numpy() * std + mean
        var = posterior.variance.detach().cpu().numpy() * (std**2)

    stddev = np.sqrt(np.clip(var, a_min=1e-12, a_max=None))
    return mu, stddev


def select_top_single_mutations(df: pd.DataFrame, top_k: int) -> List[Mutation]:
    """Return the most promising single mutations based on posterior mean."""

    singles = df[df["num_mutations"] == 1].copy()
    if singles.empty:
        raise ValueError("The dataset does not contain any single mutants.")

    singles = singles.sort_values("posterior_mean", ascending=False).head(top_k)
    return [row.mutation_list[0] for row in singles.itertuples(index=False)]


def build_candidate_library(
    wild_type: str,
    top_mutations: Sequence[Mutation],
    min_size: int,
    max_size: int,
    max_candidates: int,
    existing_sequences: Iterable[str],
) -> pd.DataFrame:
    """Generate combinatorial variants from the provided single mutations."""

    if min_size < 2:
        raise ValueError("Minimum combination size must be at least 2.")

    existing = set(existing_sequences)
    seen_sequences: set[str] = set()
    candidates: List[Tuple[str, List[Mutation]]] = []

    for size in range(min_size, max_size + 1):
        for combo in itertools.combinations(top_mutations, size):
            positions = {mut.position for mut in combo}
            if len(positions) != len(combo):
                continue

            seq = apply_mutations(wild_type, combo)
            if seq in existing or seq in seen_sequences:
                continue

            seen_sequences.add(seq)
            candidates.append((seq, list(combo)))

            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break

    if not candidates:
        raise ValueError("No candidate variants could be generated from the provided singles.")

    df = pd.DataFrame(
        {
            "sequence": [seq for seq, _ in candidates],
            "mutation_list": [combo for _, combo in candidates],
        }
    )
    df["mutations"] = df["mutation_list"].apply(mutations_to_string)
    df["num_mutations"] = df["mutation_list"].apply(len)
    return df


def greedy_d_optimal_selection(kernel_matrix: np.ndarray, noise: float, batch_size: int) -> List[int]:
    """Greedy D-optimal design selection given a kernel matrix."""

    n = kernel_matrix.shape[0]
    if batch_size >= n:
        return list(range(n))

    selected: List[int] = []
    remaining = list(range(n))

    while remaining and len(selected) < batch_size:
        best_idx = None
        best_det = -math.inf
        for idx in remaining:
            trial = selected + [idx]
            submatrix = kernel_matrix[np.ix_(trial, trial)]
            submatrix = submatrix + noise * np.eye(len(trial))
            sign, logdet = np.linalg.slogdet(submatrix)
            if sign <= 0:
                continue
            if logdet > best_det:
                best_det = logdet
                best_idx = idx

        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


def active_learning_selection(
    scores: np.ndarray,
    batch_size: int,
    rounds: int,
) -> List[Tuple[int, int]]:
    """Greedy batched acquisition selection with round annotations."""

    remaining = list(range(len(scores)))
    selections: List[Tuple[int, int]] = []

    for round_idx in range(1, rounds + 1):
        if not remaining:
            break
        order = np.argsort(scores[remaining])[::-1]
        chosen = [remaining[i] for i in order[:batch_size]]
        selections.extend((idx, round_idx) for idx in chosen)
        remaining = [idx for idx in remaining if idx not in chosen]

    return selections


def prepare_embeddings(
    embedder: ESMEmbedder,
    sequences: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    """Helper to embed sequences and convert them to tensors."""

    embeddings = embedder.embed(sequences)
    tensor = torch.tensor(embeddings, dtype=torch.float32)
    if device.type == "cuda":
        tensor = tensor.to(device=device, non_blocking=True)
    return tensor


def tokenize_sequences(
    tokenizer: Tokenizer,
    sequences: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    """Tokenize sequences and move them to the requested device."""

    tokens = tokenizer(sequences)
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens, dtype=torch.long)
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    if device.type == "cuda":
        tokens = tokens.to(device=device, non_blocking=True)
    return tokens


def make_gp_inputs(
    tokens: Optional[torch.Tensor],
    embeddings: Optional[torch.Tensor],
    zero_shot: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, ...]:
    """Package GP inputs while discarding missing components."""

    components: List[torch.Tensor] = []
    if tokens is not None:
        components.append(tokens)
    if embeddings is not None:
        components.append(embeddings)
    if zero_shot is not None:
        components.append(zero_shot)
    return tuple(components)


def compute_kernel_matrix(
    gp,
    inputs: Tuple[torch.Tensor, ...],
) -> np.ndarray:
    """Evaluate the kernel matrix for a batch of inputs."""

    with torch.no_grad():
        if isinstance(gp.covar_module, CompositeKernel):
            tokens, embeddings = inputs[0], inputs[1]
            kernel_lazy = gp.covar_module((tokens, embeddings))
        else:
            kernel_input = inputs[0]
            if (
                len(inputs) > 1
                and isinstance(kernel_input, torch.Tensor)
                and kernel_input.dtype in (torch.long, torch.int64)
            ):
                kernel_input = inputs[1]
            kernel_lazy = gp.covar_module(kernel_input)

        kernel = kernel_lazy.evaluate() if hasattr(kernel_lazy, "evaluate") else kernel_lazy

    return kernel.detach().cpu().numpy()


app = typer.Typer(help="One-liner CLI to embed DMS data, fit Kermut, and suggest variants.")


@app.command()
def run(
    tsv_path: Path = typer.Argument(..., help="Assay TSV/CSV containing sequences and fitness."),
    wild_type: str = typer.Option(..., help="Reference sequence corresponding to the assay."),
    output_dir: Path = typer.Option(Path("one_liner_outputs"), help="Where to write outputs."),
    strategy: Strategy = typer.Option(Strategy.MAX_DOE, case_sensitive=False, help="Exploration module to use."),
    target_col: str = typer.Option("fitness", help="Column with assay measurements."),
    sequence_col: str = typer.Option("mutated_sequence", help="Column holding mutated sequences."),
    mutation_col: str | None = typer.Option("mutations", help="Optional mutation code column."),
    batch_size: int = typer.Option(24, help="Number of variants to propose."),
    top_k_singles: int = typer.Option(25, help="How many single mutants to seed the library with."),
    min_mutations: int = typer.Option(2, help="Minimum number of edits per proposed variant."),
    max_mutations: int = typer.Option(4, help="Maximum number of edits per proposed variant."),
    max_candidates: int = typer.Option(500, help="Maximum size of the combinatorial library."),
    esm_model_path: Path = typer.Option(
        Path("models/esm2_t33_650M_UR50D.pt"),
        help="Path to the ESM-2 weights used for embeddings.",
    ),
    toks_per_batch: int = typer.Option(8192, help="Token budget per embedding batch."),
    lr: float = typer.Option(0.1, help="Learning rate for GP training."),
    n_steps: int = typer.Option(150, help="Number of GP optimization steps."),
    use_gpu: bool = typer.Option(False, help="Run embeddings and GP on GPU when available."),
    rounds: int = typer.Option(3, help="Active-learning rounds (Strategy.ACTIVE_LEARNING only)."),
    exploration_weight: float = typer.Option(
        1.0,
        help="Weight on predictive std when ranking candidates for active learning.",
    ),
    kernel_config: Path = typer.Option(
        DEFAULT_KERNEL_CONFIG,
        help="Hydra kernel configuration used to instantiate the GP.",
    ),
    conditional_probs: Optional[Path] = typer.Option(
        None,
        help="Optional .npy file with ProteinMPNN conditional probabilities for the wild type sequence.",
    ),
    coords_path: Optional[Path] = typer.Option(
        None,
        help="Optional .npy file with 3D coordinates (Å) for the wild type sequence.",
    ),
) -> None:
    """Run the full embedding → Kermut → suggestion pipeline."""

    output_dir.mkdir(parents=True, exist_ok=True)

    typer.secho("📥 Loading assay measurements", fg=typer.colors.CYAN)
    df = load_assay_table(
        path=tsv_path,
        wild_type=wild_type,
        sequence_col=sequence_col,
        target_col=target_col,
        mutation_col=mutation_col,
    )

    embedder = ESMEmbedder(esm_model_path, toks_per_batch=toks_per_batch, use_gpu=use_gpu)
    device = embedder.device
    tokenizer = Tokenizer()

    typer.secho("🧬 Computing embeddings for assay variants", fg=typer.colors.CYAN)
    train_embeddings = prepare_embeddings(embedder, df["sequence"].tolist(), device)

    typer.secho("🔣 Tokenising sequences", fg=typer.colors.CYAN)
    train_tokens = tokenize_sequences(tokenizer, df["sequence"].tolist(), device)

    typer.secho("📏 Standardising targets", fg=typer.colors.CYAN)
    targets, mean, std = standardize_targets(df["fitness"].to_numpy())
    if device.type == "cuda":
        targets = targets.to(device=device, non_blocking=True)

    use_structure_kernel = conditional_probs is not None or coords_path is not None
    cfg = build_config(
        kernel_config=kernel_config,
        use_gpu=use_gpu,
        lr=lr,
        n_steps=n_steps,
        embedding_dim=train_embeddings.shape[1],
        use_zero_shot=False,
        use_structure_kernel=use_structure_kernel,
    )

    gp_inputs: Dict[str, torch.Tensor] = {}
    if bool(cfg.kernel.get("use_structure_kernel", False)):
        wt_tokens = tokenizer([wild_type])
        if not isinstance(wt_tokens, torch.Tensor):
            wt_tokens = torch.tensor(wt_tokens, dtype=torch.long)
        gp_inputs["wt_sequence"] = wt_tokens.squeeze(0)

        structure_cfg = cfg.kernel.get("structure_kernel", None)
        requires_probs = False
        requires_coords = False
        if structure_cfg is not None:
            requires_probs = bool(
                getattr(structure_cfg, "use_site_comparison", False)
                or getattr(structure_cfg, "use_mutation_comparison", False)
            )
            requires_coords = bool(getattr(structure_cfg, "use_distance_comparison", False))

        if requires_probs or conditional_probs is not None:
            if conditional_probs is None:
                raise ValueError(
                    "Structure kernel requested but no conditional probabilities were provided. "
                    "Supply --conditional-probs with a .npy file."
                )
            probs = np.load(conditional_probs)
            gp_inputs["conditional_probs"] = torch.tensor(probs, dtype=torch.float32)

        if requires_coords or coords_path is not None:
            if coords_path is None:
                raise ValueError(
                    "Structure kernel requested but no coordinate file was provided. "
                    "Supply --coords-path with a .npy file."
                )
            coords = np.load(coords_path)
            gp_inputs["coords"] = torch.tensor(coords, dtype=torch.float32)

    typer.secho("📈 Training Kermut Gaussian process", fg=typer.colors.CYAN)
    train_inputs = make_gp_inputs(train_tokens, train_embeddings, None)
    gp, likelihood = train_gp(train_inputs=train_inputs, targets=targets, cfg=cfg, gp_inputs=gp_inputs)

    typer.secho("🔍 Evaluating posterior on training data", fg=typer.colors.CYAN)
    mu_train, sigma_train = predict_posterior(gp=gp, likelihood=likelihood, inputs=train_inputs, mean=mean, std=std)
    df["posterior_mean"] = mu_train
    df["posterior_std"] = sigma_train

    training_out = output_dir / "training_posterior.csv"
    df.drop(columns=["mutation_list"]).to_csv(training_out, index=False)
    typer.secho(f"✅ Stored training posterior at {training_out}", fg=typer.colors.GREEN)

    typer.secho("🧪 Building combinatorial candidate library", fg=typer.colors.CYAN)
    top_mutations = select_top_single_mutations(df, top_k=top_k_singles)
    candidates = build_candidate_library(
        wild_type=wild_type,
        top_mutations=top_mutations,
        min_size=min_mutations,
        max_size=max_mutations,
        max_candidates=max_candidates,
        existing_sequences=df["sequence"].tolist(),
    )

    typer.secho("🧬 Computing embeddings for candidate variants", fg=typer.colors.CYAN)
    cand_embeddings = prepare_embeddings(embedder, candidates["sequence"].tolist(), device)
    cand_tokens = tokenize_sequences(tokenizer, candidates["sequence"].tolist(), device)
    candidate_inputs = make_gp_inputs(cand_tokens, cand_embeddings, None)

    typer.secho("📊 Scoring candidates with Kermut", fg=typer.colors.CYAN)
    mu_cand, sigma_cand = predict_posterior(
        gp=gp,
        likelihood=likelihood,
        inputs=candidate_inputs,
        mean=mean,
        std=std,
    )
    candidates["posterior_mean"] = mu_cand
    candidates["posterior_std"] = sigma_cand

    suggestions_path = output_dir / f"suggestions_{strategy.value}.csv"

    if strategy is Strategy.MAX_DOE:
        typer.secho("🧮 Running greedy D-optimal selection", fg=typer.colors.CYAN)
        kernel = compute_kernel_matrix(gp, candidate_inputs)
        noise = float(likelihood.noise.detach().cpu())
        selected_idx = greedy_d_optimal_selection(kernel, noise=noise, batch_size=batch_size)
        selected = candidates.iloc[selected_idx].copy()
        selected["selection_rank"] = range(1, len(selected) + 1)
        selected.to_csv(suggestions_path, index=False)
        typer.secho(f"✅ Saved D-optimal batch to {suggestions_path}", fg=typer.colors.GREEN)
    else:
        typer.secho("🔁 Running active-learning style acquisition", fg=typer.colors.CYAN)
        acquisition = mu_cand + exploration_weight * sigma_cand
        selections = active_learning_selection(acquisition, batch_size=batch_size, rounds=rounds)
        if not selections:
            raise RuntimeError("Active-learning strategy could not select any candidates.")
        rows, rounds_out = zip(*selections)
        selected = candidates.iloc[list(rows)].copy()
        selected["round"] = list(rounds_out)
        order = np.argsort(acquisition[list(rows)])[::-1]
        selected = selected.iloc[order].reset_index(drop=True)
        selected["selection_rank"] = range(1, len(selected) + 1)
        selected.to_csv(suggestions_path, index=False)
        typer.secho(f"✅ Saved active-learning batches to {suggestions_path}", fg=typer.colors.GREEN)

    typer.secho("🎉 Pipeline finished", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
