"""huggingface_example: A Flower / Hugging Face app."""

import warnings
import torch, os, gc
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
from opacus import PrivacyEngine
# from transformers import (
#     AutoTokenizer,
#     # DataCollatorWithPadding,
#     DataCollatorForTokenClassification,
#     AutoModelForTokenClassification,
# )
# from transformers import logging
from flwr.client.mod import LocalDpMod     # <- does clipping + Gaussian noise
import logging
from torch.optim import AdamW, Adafactor
from huggingface_example.task import (
    train,
    test,
    load_data,
    set_params,
    get_params,
    get_model,
)
from torch.nn import DataParallel
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure PyTorch memory settings
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for matmul
torch.backends.cudnn.allow_tf32 = True        # Enable TF32 for convolutions
torch.set_float32_matmul_precision('medium')  # Balance speed/accuracy
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3, 4, 5"
# To mute warnings reminding that we need to train the model to a downstream task
# This is something this example does.
# logging.set_verbosity_error()

# Keep each Ray actor to a single core
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Disable the HF tokenizer’s own parallelism - it defeats the point above
os.environ["TOKENIZERS_PARALLELISM"] = "false"

CLIP_NORM = 1.0        # L2 clipping radius
EPSILON   = 8.0        # total privacy budget
DELTA     = 1e-5       # failure probability

# sensitivity is usually 1.0 if you clip *gradients*; keep the default
local_dp_obj = LocalDpMod(clipping_norm=CLIP_NORM,
                    sensitivity=1.0,
                    epsilon=EPSILON,
                    delta=DELTA)


# Flower client
class IMDBClient_(NumPyClient):
    def __init__(self, model_name, trainloader, testloader, local_epochs) -> None:
        
        torch.set_num_threads(1)
       
        self.device = torch.device("cuda")
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.net = get_model(model_name)
        self.net.to(self.device)

    def fit(self, parameters, config) -> tuple[list, int, dict]:
        set_params(self.net, parameters)
        train_loss, train_time  = train(self.net, self.trainloader, epochs= self.local_epochs, device=self.device)
        print(f"[Client] Training time: {train_time:.2f} seconds")
        return (get_params(self.net), len(self.trainloader), {
            "avg_train_loss": train_loss,
            "train_time": train_time,  # ⏱️ added metric
        })

    def evaluate(self, parameters, config) -> tuple[float, int, dict[str, float]]:
        set_params(self.net, parameters)
        loss, f1 = test(self.net, self.testloader, device=self.device)
        return float(loss), len(self.testloader), {"f1": float(f1)}

# Flower client
class IMDBClient(NumPyClient):
    def __init__(self, model_name, trainloader, testloader, local_epochs, 
                 target_delta,
                 noise_multiplier,
                 max_grad_norm,) -> None:# Configure PyTorch memory settings
        
        torch.set_num_threads(1)
        # torch.set_num_interop_threads(1)
        # torch.cuda.set_device(0)        # index 0 in _local_ CUDA_VISIBLE_DEVICES
        
        self.device = torch.device("cuda") 
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.net = get_model(model_name)
        # self.net = DataParallel(self.net)
        self.net.to(self.device)
        
        self.target_delta = target_delta
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm

        # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        # os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15"
        
        # --- ADD THIS VERIFICATION BLOCK ---
        # if self.device.type == "cuda":
        #     gpu_count = torch.cuda.device_count()
        #     print(f"[Client] Visible GPUs: {gpu_count}")
        #     for i in range(gpu_count):
        #         print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        # ------------------------------------
        # torch.cuda.set_per_process_memory_fraction(0.9)  # Leave 10% buffer

    def fit(self, parameters, config) -> tuple[list, int, dict]:
        torch.cuda.empty_cache()
        set_params(self.net, parameters)
        optimizer = AdamW(self.net.parameters(), lr=1e-5)
        # optimizer = Adafactor(
        #                 self.net.parameters(), 
        #                 lr=1e-5,
        #             )
        self.net.train()
        privacy_engine = PrivacyEngine(secure_mode = False)
        (
            self.net,
            optimizer,
            self.trainloader,
        ) = privacy_engine.make_private(
            module=self.net,
            optimizer = optimizer,
            data_loader = self.trainloader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )
        # model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        #                                     module=model,
        #                                     optimizer=optimizer,
        #                                     data_loader=train_loader,
        #                                     epochs=EPOCHS,
        #                                     target_epsilon=EPSILON,
        #                                     target_delta=DELTA,
        #                                     max_grad_norm=MAX_GRAD_NORM,)


        train_loss, train_time, epsilon  = train(self.net, self.trainloader,
                                                 epochs = self.local_epochs, 
                                                 device = self.device, 
                                                 privacy_engine = privacy_engine, 
                                                 optimizer = optimizer, 
                                                 target_delta = self.target_delta,
                                                )
        if epsilon is not None:
            print(f"Epsilon value for delta={self.target_delta} is {epsilon:.2f}")
        else:
            print("Epsilon value not available.") 
        
        print(f"[Client] Training time: {train_time:.2f} seconds")
        return (get_params(self.net), len(self.trainloader.dataset), {
            "avg_train_loss": train_loss,
            "train_time": train_time,  # ⏱️ added metric
        })

        # --- add right below the return statement ---
        # self.net.to("cpu")          # release all tensors from device
        del optimizer, privacy_engine
        torch.cuda.empty_cache()    # give the memory back to CUDA driver
        gc.collect()    

    def evaluate(self, parameters, config) -> tuple[float, int, dict[str, float]]:
        set_params(self.net, parameters)
        loss, f1 = test(self.net, self.testloader, device=self.device)
        return float(loss), len(self.testloader.dataset), {"f1": float(f1)}

def client_fn(context: Context) -> Client:
    """Construct a Client that will be run in a ClientApp."""
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    local_epochs = context.run_config["local-epochs"]
    # Read the run config to get settings to configure the Client
    model_name = context.run_config["model-name"]
    noise_multiplier = 1.0 if partition_id % 2 == 0 else 1.5
    
    trainloader, testloader = load_data(partition_id, num_partitions, model_name, cache_dir = '../cache_dir')

    return IMDBClient(model_name, trainloader, testloader, local_epochs, 
                      context.run_config["target-delta"],
                      noise_multiplier,
                      context.run_config["max-grad-norm"]).to_client()

app = ClientApp(client_fn=client_fn)
