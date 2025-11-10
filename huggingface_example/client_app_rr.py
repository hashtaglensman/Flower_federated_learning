import warnings
import torch, os, gc
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
import logging
from flwr.client.mod import LocalDpMod 
from torch.optim import AdamW, Adafactor
from huggingface_example.task import (
     train_wo_dp,
     test,
     load_data,
     set_params,
     get_params,
     get_model,
     randomized_response_on_deltas,
 )

from torch.nn import DataParallel
warnings.filterwarnings("ignore", category=FutureWarning)

class IMDBClient(NumPyClient):
    def __init__(self, model_name, trainloader, testloader, local_epochs,  rr_epsilon: float | None,
              rr_flip_prob: float | None,
              max_grad_norm,) -> None:# Configure PyTorch memory settings
        
        torch.set_num_threads(1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.net = get_model(model_name).to(self.device)
        
        self.rr_epsilon = rr_epsilon
        self.rr_flip_prob = rr_flip_prob
        self.max_grad_norm = max_grad_norm
        
    def fit(self, parameters, config) -> tuple[list, int, dict]:
        torch.cuda.empty_cache()
        # Keep a copy of incoming global weights
        set_params(self.net, parameters)
        base_params = get_params(self.net)
        optimizer = AdamW(self.net.parameters(), lr=1e-5) 
        self.net.train()

         # --- Standard local training (no Opacus; we’ll privatize the *update*) ---
        train_losses, train_time = train_wo_dp(
             self.net,
             self.trainloader,
             epochs=self.local_epochs,
             device=self.device,
             optimizer=optimizer,
             grad_clip_norm=self.max_grad_norm,
         )

        
        # --- Compute deltas and apply Randomized Response on *signs* -------------
        new_params = get_params(self.net)
        deltas = [n - b for n, b in zip(new_params, base_params)]
        deltas_rr = randomized_response_on_deltas(
                                        deltas,
                                        flip_prob=self.rr_flip_prob,
                                        epsilon=self.rr_epsilon,
                                        device=self.device,
                                        )
        # Rebuild sent parameters = base + perturbed_deltas
        sent_params = [b + d for b, d in zip(base_params, deltas_rr)]
        
        # --- Return parameters and metrics ---------------------------------------
        result = (
                    sent_params,
                    len(self.trainloader.dataset),
                     {
                        "avg_train_loss": float(sum(train_losses) / max(1, len(train_losses))),
                        "train_time": train_time,
                        "rr_flip_prob": float(self.rr_flip_prob) if self.rr_flip_prob is not None else None,
                        "rr_epsilon": float(self.rr_epsilon) if self.rr_epsilon is not None else None,
                     },
                )
        
        return result
        del optimizer
        torch.cuda.empty_cache()
        gc.collect() 
        
    def evaluate(self, parameters, config) -> tuple[float, int, dict[str, float]]:
        set_params(self.net, parameters)
        loss, f1 = test(self.net, self.testloader, device=self.device)
        return float(loss), len(self.testloader.dataset), {"f1": float(f1)}

def client_fn(context: Context) -> Client:
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    local_epochs = context.run_config["local-epochs"]
    model_name = context.run_config["model-name"]
    noise_multiplier = 0.05 if partition_id % 2 == 0 else 0.5
    # RR configuration from run_config (either ε or direct flip prob)
    rr_epsilon = context.run_config.get("rr-epsilon", None)
    rr_flip_prob = context.run_config.get("rr-flip-prob", None)
    trainloader, testloader = load_data(partition_id, num_partitions, model_name, cache_dir = '../cache_dir')
    return IMDBClient(
                model_name, trainloader, testloader, local_epochs,
                rr_epsilon, rr_flip_prob,
                context.run_config["max-grad-norm"]
            ).to_client()

app = ClientApp(client_fn=client_fn)
