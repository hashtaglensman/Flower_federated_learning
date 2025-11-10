"""huggingface_example: A Flower / Hugging Face app."""

import warnings
import torch, os
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
from flwr.common import Code, Status, FitIns, ndarrays_to_parameters
from torch.optim import AdamW
from transformers import logging
import gc

from huggingface_example.task import (
    train,
    test,
    load_data,
    set_params,
    get_params,
    get_model,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# To mute warnings reminding that we need to train the model to a downstream task
# This is something this example does.
# logging.set_verbosity_error()
# --- PyTorch / system settings ---
torch.backends.cuda.matmul.allow_tf32 = True   # Enable TF32 for matmul
torch.backends.cudnn.allow_tf32 = True         # Enable TF32 for convolutions
torch.set_float32_matmul_precision("medium")   # Balance speed/accuracy
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Flower client
class IMDBClient(NumPyClient):
    def __init__(self, model_name, trainloader, testloader, local_epochs) -> None:
        self.device = torch.device("cuda")
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.net = get_model(model_name)
        self.net.to(self.device)

    def fit(self, parameters, config):        # 👈 return type #config
        set_params(self.net, parameters)
        optimizer = AdamW(self.net.parameters(), lr=1e-5)
        train_loss, train_time = train(
            self.net, self.trainloader,
            epochs=self.local_epochs,
            device=self.device,
            optimizer=optimizer,
        )
        print(f"[Client] Training time: {train_time:.2f} seconds")
        # Return updated weights and number of examples
        num_examples = (
            len(self.trainloader.dataset)
            if hasattr(self.trainloader, "dataset")
            else sum(len(b[0]) for b in self.trainloader)  # fallback
        )
        
        del optimizer
        # Clean up
        torch.cuda.empty_cache() if self.device.type == "cuda" else None
        gc.collect()
    
        return get_params(self.net), num_examples, {
            "avg_train_loss": train_loss,
            "train_time": train_time,
        }


    def evaluate(self, parameters, config):
        set_params(self.net, parameters)
        loss, f1 = test(self.net, self.testloader, device=self.device)
        num_examples = (
            len(self.testloader.dataset)
            if hasattr(self.testloader, "dataset")
            else sum(len(b[0]) for b in self.testloader)
        )
        return float(loss), num_examples, {"f1": float(f1)}



def client_fn(context: Context) -> Client:
    """Construct a Client that will be run in a ClientApp."""
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    local_epochs = context.run_config["local-epochs"]
    # Read the run config to get settings to configure the Client
    model_name = context.run_config["model-name"]
    
    trainloader, testloader = load_data(partition_id, num_partitions, model_name, cache_dir = '../cache_dir')

    return IMDBClient(model_name, trainloader, testloader, local_epochs).to_client()


app = ClientApp(client_fn=client_fn)
