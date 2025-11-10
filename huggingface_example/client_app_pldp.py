"""huggingface_example: A Flower / Hugging Face app (client)."""

import warnings
import math
import time, gc, os
from typing import Dict, Tuple, List, Any, Optional
import torch
from torch.optim import AdamW
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
# Opacus imports (we only need the PrivacyEngine handle and a noise generator base)
from huggingface_example.task import (
    train_wo_dp,
    test,
    load_data,
    set_params,
    get_params,
    get_model,
)

warnings.filterwarnings("ignore", category=FutureWarning)

torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for matmul
torch.backends.cudnn.allow_tf32 = True        # Enable TF32 for convolutions
torch.set_float32_matmul_precision('medium')  # Balance speed/accuracy

# Keep each Ray actor to a single core
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# # Disable the HF tokenizer’s own parallelism - it defeats the point above
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


# -------------------------------
# Laplace noise helper
# -------------------------------
def add_laplace_noise(tensor, scale, device):
    dist = torch.distributions.Laplace(loc=0.0, scale=scale)
    noise = dist.sample(tensor.shape).to(device)
    return tensor + noise

# ----------------------------
# Helpers to resolve per-client epsilon
# ----------------------------
def resolve_client_epsilon(context: Context, partition_id: int) -> Tuple[float, str]:
    """
    Decide this client's ε (privacy budget) and return (epsilon, source_tag).

    Priority:
      1) context.run_config["personalized-epsilon-map"][str(partition_id)]
      2) context.node_config["epsilon"] (float) or node_config["privacy-level"] in {high, medium, low}
      3) Fallback deterministic pattern by partition id
    """
    # 1) Explicit mapping from run_config
    eps_map = context.run_config.get("personalized-epsilon-map", {})
    if isinstance(eps_map, dict) and str(partition_id) in eps_map:
        try:
            return float(eps_map[str(partition_id)]), "run_config_map"
        except Exception:
            pass  # fall through if malformed

    # 2a) Node-level numeric epsilon
    if "epsilon" in context.node_config:
        try:
            return float(context.node_config["epsilon"]), "node_config_epsilon"
        except Exception:
            pass

    # 2b) Node-level privacy level
    lvl = str(context.node_config.get("privacy-level", "")).lower()
    lvl2eps = {"high": 2.0, "medium": 6.0, "low": 10.0}
    if lvl in lvl2eps:
        return lvl2eps[lvl], f"node_privacy_level_{lvl}"

    # 3) Deterministic fallback pattern
    #   more sensitive every 3rd client → ε=2.0, next → ε=6.0, next → ε=10.0
    r = partition_id % 3
    return (2.0 if r == 0 else 6.0 if r == 1 else 10.0), "fallback_pattern"


def laplace_noise_multiplier_from_epsilon(epsilon: float) -> float:
    """
    For Laplace mechanism with L1 sensitivity S≈max_grad_norm (after clipping),
    scale b = S / ε ⇒ noise_multiplier = 1 / ε (so scale = max_grad_norm / ε).
    """
    epsilon = max(float(epsilon), 1e-6)
    return 1.0 / epsilon
    
def _clip_deltas_by_global_l1(deltas: List[torch.Tensor], max_l1_norm: float) -> List[torch.Tensor]:
    """Globally clip a list of tensors by L1 norm (Laplace sensitivity bound)."""
    eps = 1e-12
    # compute global L1 over the whole update (sum of |t| across all tensors)
    total_l1 = 0.0
    for t in deltas:
        total_l1 += t.abs().sum().item()
    if total_l1 <= max_l1_norm:
        return deltas
    scale = max_l1_norm / (total_l1 + eps)
    return [t * scale for t in deltas]

# ----------------------------
# Flower NumPyClient
# ----------------------------
class IMDBClient(NumPyClient):
    def __init__(
        self,
        model_name: str,
        trainloader,
        testloader,
        local_epochs: int,
        target_delta: float,
        noise_multiplier: float,
        max_grad_norm: float,
        epsilon_budget: float,
        epsilon_source: str,
    ) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = get_model(model_name).to(self.device)

        self.model_name = model_name
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = int(local_epochs)
        self.target_delta = float(target_delta)

        self.noise_multiplier = float(noise_multiplier)
        self.max_grad_norm = float(max_grad_norm)

        self.epsilon_budget = float(epsilon_budget)
        self.epsilon_source = str(epsilon_source)

    # --- Flower API ---
    def get_parameters(self, config: Dict[str, Any]) -> List[torch.Tensor]:
        return get_params(self.net)

    def set_parameters(self, parameters: List[torch.Tensor]) -> None:
        set_params(self.net, parameters)

    def fit(self, parameters: List[torch.Tensor], config: Dict[str, Any]):
        torch.cuda.empty_cache()
        
        # 1) Load server params and snapshot as the "base" to compute a delta later
        self.set_parameters(parameters)
        base_params = [torch.tensor(p, device=self.device, dtype=torch.float32) for p in parameters]
        
        # 2) Standard local training (no DP in the inner loop)
        optimizer = AdamW(self.net.parameters(), lr=4e-4)
        self.net.train()
        print(
            f"[Client {config.get('cid','?')}] Personalized LDP → "
            f"ε={self.epsilon_budget:.3f} (source={self.epsilon_source}), "
            f"noise_multiplier={self.noise_multiplier:.6f}, "
            f"max_grad_norm={self.max_grad_norm:.6f}"
            )
        
        epoch_losses, train_time = train_wo_dp(
        self.net,
        self.trainloader,
        epochs=self.local_epochs,
        device=self.device,
        optimizer=optimizer,
        grad_clip_norm=self.max_grad_norm,  # keeps gradients sane during training
        )
        
        # 3) Compute raw delta (local model - base)
        new_params = [torch.tensor(p, device=self.device, dtype=torch.float32) for p in get_params(self.net)]
        deltas = [n - b for n, b in zip(new_params, base_params)]
        
        # 4) Global L1 clip (Laplace sensitivity bound)
        deltas = _clip_deltas_by_global_l1(deltas, max_l1_norm=float(self.max_grad_norm))
        
        # 5) Add Laplace noise with scale = S/ε (S ≈ max_grad_norm after clipping)
        eps = max(float(self.epsilon_budget), 1e-6)
        laplace_scale = float(self.max_grad_norm) / eps
        noisy_deltas = [add_laplace_noise(d, laplace_scale, self.device) for d in deltas]
        
        # 6) Build perturbed parameters and set them on the model we return
        perturbed_params = [ (b + nd).detach().cpu().numpy() for b, nd in zip(base_params, noisy_deltas) ]
        set_params(self.net, perturbed_params)
        
        # 7) Prepare metrics and return
        avg_loss = float(sum(epoch_losses) / max(1, len(epoch_losses))) if epoch_losses else float("nan")
        metrics = {
        "avg_train_loss": avg_loss,
        "train_time": float(train_time),
        "epsilon_budget": float(self.epsilon_budget),
        "epsilon_source": self.epsilon_source,
        "noise_multiplier": float(self.noise_multiplier),
        "laplace_scale": laplace_scale,
        "clip_L1_bound": float(self.max_grad_norm),
        "pldp_applied": True,
        }
        
        return get_params(self.net), len(self.trainloader.dataset), metrics
        del optimizer
        torch.cuda.empty_cache()
        gc.collect() 
        
    def evaluate(self, parameters: List[torch.Tensor], config: Dict[str, Any]):
        self.set_parameters(parameters)
        loss, f1 = test(self.net, self.testloader, device=self.device)
        metrics = {"f1": float(f1)}
        return float(loss), len(self.testloader.dataset), metrics
# ----------------------------
# Client factory for Flower ClientApp
# ----------------------------
def client_fn(context: Context) -> Client:
    """
    Construct a client instance using the provided Flower Context.
    - Applies Personalized LDP: picks a per-client ε and converts it to Laplace noise.
    """
    partition_id = (context.node_config["partition-id"])
    num_partitions = (context.node_config["num-partitions"])
    local_epochs = (context.run_config["local-epochs"])
    model_name = context.run_config["model-name"]

    # Personalized ε for this client
    epsilon_budget, eps_src = resolve_client_epsilon(context, partition_id)

    # Convert ε → Laplace noise multiplier (scale = max_grad_norm / ε)
    max_grad_norm = float(context.run_config["max-grad-norm"])
    noise_multiplier = laplace_noise_multiplier_from_epsilon(epsilon_budget)

    # Load local data shard
    trainloader, testloader = load_data(
        partition_id,
        num_partitions,
        model_name,
        cache_dir="../cache_dir",
    )

    return IMDBClient(
        model_name=model_name,
        trainloader=trainloader,
        testloader=testloader,
        local_epochs=local_epochs,
        target_delta=float(context.run_config["target-delta"]),
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        epsilon_budget=epsilon_budget,
        epsilon_source=eps_src,
    ).to_client()


# Flower entrypoint
app = ClientApp(client_fn=client_fn)
