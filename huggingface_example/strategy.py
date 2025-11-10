"""pytorch-example: A Flower / PyTorch app (custom FedAvg strategy)."""

import atexit
import json, os
from logging import INFO
from pathlib import Path
from time import time
import torch
import wandb
from flwr.common import logger, parameters_to_ndarrays
from flwr.common.typing import UserConfig
from flwr.server.strategy import FedAvg

from huggingface_example.task import create_run_dir, get_model, set_params

PROJECT_NAME = "FLOWER-BERT-NCBI-disease_3nodes_3g_24c_wCPUbinding_SegAcc"
MODEL_NAME = "bert-base-uncased"

class CustomFedAvg(FedAvg):
    """FedAvg plus: JSON metrics dump, checkpoint-on-best, W&B logging."""

    # --------------------------------------------------------------------- #
    #  Initialisation
    # --------------------------------------------------------------------- #
    def __init__(self, run_config: UserConfig, use_wandb: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Where to save artefacts for this run
        self.save_path, self.run_dir = create_run_dir(run_config)
        self.results_path = Path(self.save_path) / "results.json"

        self.use_wandb = use_wandb
        # if use_wandb:
        self._init_wandb_project()

        # Tracking helpers
        self.best_f1_so_far = 0.0
        self.results: dict[str, list[dict]] = {}
        self._round_start_time: float | None = None
        self.total_server_time: float = 0.0
        self._sim_start_wall: float = time()

        # Register clean-up callback *before* interpreter shutdown
        atexit.register(self._on_exit)

        # Opt-in to the new Weights & Biases backend (avoids future warning)
        wandb.require("core")
        self.server_core = None
        if hasattr(os, 'sched_getaffinity'):
            self.server_core = os.sched_getaffinity(0)
            print(f"Server CPU affinity confirmed: {self.server_core}")

    def _init_wandb_project(self) -> None:
        wandb.init(project=PROJECT_NAME, name=f"{self.run_dir}-ServerApp")

    # --------------------------------------------------------------------- #
    #  Generic helpers
    # --------------------------------------------------------------------- #
    def _store_results(self, tag: str, results_dict: dict) -> None:
        """Update in-memory metrics and persist to <run_dir>/results.json."""
        self.results.setdefault(tag, []).append(results_dict)
        # Overwrite (JSON is tiny); for larger jobs consider incremental writes
        with open(self.results_path, "w", encoding="utf-8") as fp:
            json.dump(self.results, fp, indent=2)

    def store_results_and_log(
        self, server_round: int, tag: str, results_dict: dict
    ) -> None:
        """Helper: save + (optionally) W&B-log a dict of metrics."""
        self._store_results(tag, {"round": server_round, **results_dict})

        if self.use_wandb:
            wandb.log({f"{tag}/{k}": v for k, v in results_dict.items()},
                      step=server_round)

    # --------------------------------------------------------------------- #
    #  Best-model checkpointing
    # --------------------------------------------------------------------- #
    def _update_best_f1(self, round: int, f1: float, parameters) -> None:
        if f1 <= self.best_f1_so_far:
            return

        self.best_f1_so_far = f1
        logger.log(INFO, "💡 New best global model found: %f", f1)

        ndarrays = parameters_to_ndarrays(parameters)
        model = get_model(MODEL_NAME)
        set_params(model, ndarrays)

        file_name = f"model_state_f1_{f1:.4f}_round_{round}.pth"
        torch.save(model.state_dict(), Path(self.save_path) / file_name)

    # --------------------------------------------------------------------- #
    #  Strategy hooks (timing + metrics)
    # --------------------------------------------------------------------- #
    def configure_fit(self, server_round, parameters, client_manager):
        self._round_start_time = time()
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)

        round_time = time() - (self._round_start_time or time())
        self.total_server_time += round_time

        self.store_results_and_log(
            server_round,
            tag="server_time",
            results_dict={
                "round_time_sec": round_time,
                "cumulative_server_time_sec": self.total_server_time,
            },
        )
        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        self.store_results_and_log(
            server_round,
            tag="federated_evaluate",
            results_dict={"federated_evaluate_loss": loss, **metrics},
        )
        return loss, metrics

    # --------------------------------------------------------------------- #
    #  Clean shutdown callback (replaces fragile __del__)
    # --------------------------------------------------------------------- #
    def _on_exit(self) -> None:
        """Runs while modules are still alive, so built-ins are safe."""
        total_wall = time() - self._sim_start_wall
        try:
            self._store_results(
                tag="server_time",
                results_dict={"total_wall_time_sec": total_wall},
            )
            logger.log(INFO, "🏁  Total server runtime: %.2f s", total_wall)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to record final runtime: %s", exc)
