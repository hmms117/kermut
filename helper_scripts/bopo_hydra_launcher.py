"""Hydra front-end for running the BOPO + Kermut policy learner via Typer."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from hydra import compose, initialize
from omegaconf import OmegaConf

from helper_scripts import bopo_policy_learning

app = typer.Typer(help="Run BOPO policy learning using a Hydra configuration file.")

_DEFAULT_CONFIG_PATH = Path("kermut/hydra_configs")
_DEFAULT_CONFIG_NAME = "bopo_multiobjective_example"


def _normalise_site_pool(values: Optional[List[int]]) -> Optional[str]:
    """Return a comma-separated site pool string as expected by Typer CLI."""

    if not values:
        return None
    return ",".join(str(int(value)) for value in values)


@app.command()
def run(
    *,
    config_name: str = typer.Option(
        _DEFAULT_CONFIG_NAME,
        help="Hydra configuration to load (relative to --config-path).",
    ),
    config_path: Path = typer.Option(
        _DEFAULT_CONFIG_PATH,
        help="Directory that stores Hydra configuration files.",
    ),
    override: Optional[List[str]] = typer.Option(
        None,
        "--override",
        "-o",
        help="Optional Hydra-style overrides, e.g. --override data=/tmp/assay.tsv",
    ),
) -> None:
    """Execute BOPO policy learning with parameters provided by a Hydra config."""

    overrides = list(override) if override is not None else []
    try:
        with initialize(version_base=None, config_path=str(config_path)):
            cfg = compose(config_name=config_name, overrides=overrides)
    except FileNotFoundError as exc:  # pragma: no cover - hydra compose error path
        typer.secho(f"Unable to locate Hydra config '{config_name}' in {config_path}.", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        typer.secho("Hydra configuration must be a mapping of parameters.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    required_keys = ["data", "checkpoint", "objectives", "directions", "weights"]
    missing = [key for key in required_keys if key not in cfg_dict]
    if missing:
        typer.secho(
            "Missing required keys in Hydra config: " + ", ".join(missing),
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    data_path = Path(str(cfg_dict["data"]))
    ckpt_path = Path(str(cfg_dict["checkpoint"]))
    objectives = list(cfg_dict["objectives"])
    directions = list(cfg_dict["directions"])
    weights = [float(value) for value in cfg_dict["weights"]]

    site_pool_values = cfg_dict.get("site_pool")
    site_pool = None
    if site_pool_values is not None:
        if isinstance(site_pool_values, (list, tuple)):
            site_pool = _normalise_site_pool(list(site_pool_values))
        else:
            typer.secho("site_pool must be a list of integer positions.", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    extra_kwargs = {
        "epochs": int(cfg_dict.get("epochs", 30)),
        "B": int(cfg_dict.get("B", 256)),
        "K": int(cfg_dict.get("K", 16)),
        "batch_size": int(cfg_dict.get("batch_size", 96)),
        "min_muts": int(cfg_dict.get("min_muts", 2)),
        "max_muts": int(cfg_dict.get("max_muts", 11)),
        "site_pool": site_pool,
        "lr": float(cfg_dict.get("lr", 5e-2)),
        "seed": int(cfg_dict.get("seed", 123)),
        "device": str(cfg_dict.get("device", "cuda")),
        "base_seq": cfg_dict.get("base_seq"),
        "out_csv": Path(str(cfg_dict.get("out_csv", "bopo_next_batch.csv"))),
    }

    typer.secho(
        f"Running BOPO policy learning with config '{config_name}' from {config_path}",
        fg=typer.colors.BLUE,
    )

    try:
        bopo_policy_learning.suggest(
            data=data_path,
            ckpt=ckpt_path,
            objectives=objectives,
            directions=directions,
            weights=weights,
            **extra_kwargs,
        )
    except typer.BadParameter as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":  # pragma: no cover - Typer entry point
    app()
