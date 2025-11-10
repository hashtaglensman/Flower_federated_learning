"""huggingface_example: A Flower / Hugging Face app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from huggingface_example.task import get_params, get_model, set_params, test
from huggingface_example.strategy import CustomFedAvg
from torch.utils.data import DataLoader
import torch
from datasets import load_dataset
from time import time
import logging
# Opacus logger seems to change the flwr logger to DEBUG level. Set back to INFO
logging.getLogger("flwr").setLevel(logging.INFO)
from typing import List, Tuple
import os, psutil
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# # Initialize TF first without GPU
# import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')

from transformers import (
    AutoTokenizer,
    # DataCollatorWithPadding,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
)
# # ─── server_app.py ──────────────────────────────────────────────────────────────
# import os
# from pathlib import Path

# def _pin_to_cores(cores: str = "0", mems: str = "0", cg_name: str = "server_app"):
#     """
#     Place *this* process inside a dedicated cpuset cgroup restricted to the
#     given CPU cores (and NUMA nodes).

#     Works on both the legacy cpuset hierarchy (v1) and the unified hierarchy
#     (v2).  Falls back to sched_setaffinity when we lack permission.
#     """
#     CGROOT = Path("/sys/fs/cgroup")
#     is_v2  = (CGROOT / "cgroup.controllers").exists()

#     try:
#         if is_v2:
#             cg_path = CGROOT / cg_name
#             cg_path.mkdir(exist_ok=True)

#             # Make sure the cpuset controller is active under /
#             with (CGROOT / "cgroup.subtree_control").open("r+") as f:
#                 controllers = f.read()
#                 if "cpuset" not in controllers:
#                     f.seek(0, os.SEEK_END)
#                     f.write("+cpuset\n")

#             # Limit CPUs / memory nodes and move our PID
#             (cg_path / "cpuset.cpus").write_text(cores)
#             (cg_path / "cpuset.mems").write_text(mems)
#             (cg_path / "cgroup.procs").write_text(str(os.getpid()))

#         else:  # cpuset v1
#             cg_path = CGROOT / "cpuset" / cg_name
#             cg_path.mkdir(parents=True, exist_ok=True)
#             (cg_path / "cpuset.cpus").write_text(cores)
#             (cg_path / "cpuset.mems").write_text(mems)
#             (cg_path / "tasks").write_text(str(os.getpid()))

#         print(f"[cgroup] pinned to CPU(s) {cores} (mem nodes {mems}) via {cg_path}")

#     except PermissionError:
#         # Not root?  Use the best we can: sched_setaffinity
#         os.sched_setaffinity(0, {int(c) for c in cores.split(",")})
#         print(f"[affinity] pinned to CPU(s) {cores} with sched_setaffinity")

#     except OSError as exc:
#         # Kernel without cpuset?  Fall back too.
#         os.sched_setaffinity(0, {int(c) for c in cores.split(",")})
#         print(f"[fallback] cpuset unavailable ({exc}); used sched_setaffinity")

# # Pin as early as possible.  Honour an env-var so you don’t hard-code cores
# _pin_to_cores(os.getenv("SERVER_CPU_CORES", "3"))
# # ────────────────────────────────────────────────────────────────────────────────


# mo_name = "bert-base-uncased"#context.run_config["model-name"]
cache_dir = '../cache_dir'

def data_prep(model_name, split_train):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512, cache_dir = cache_dir,  force_download=True)
    def tokenize_and_align_labels(batch):
        """Tokenise and map word-level NER tags to word-piece indices."""
        tokenised = tokenizer(
            batch["tokens"],
            truncation=True,
            is_split_into_words=True,
        )
        
        aligned_labels = []
        for i in range(len(batch["tokens"])):              # iterate over each example
            word_ids   = tokenised.word_ids(batch_index=i)
            lbl_ids    = []
            prev_word  = None
        
            for word_idx in word_ids:
                if word_idx is None:                      # special / padding token
                    lbl_ids.append(-100)
                elif word_idx != prev_word:               # first sub-token of a word
                    lbl_ids.append(batch["ner_tags"][i][word_idx])
                else:                                     # subsequent sub-token
                    lbl_ids.append(batch["ner_tags"][i][word_idx])
                prev_word = word_idx
        
            aligned_labels.append(lbl_ids)
        
        tokenised["labels"] = aligned_labels
        return tokenised
  
    # 3️⃣  Apply the mapping and drop unused columns --------------------------------
    test = split_train.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=["tokens", "ner_tags", "id"],  # keep only model inputs
    )

    # 4️⃣  Build collator and PyTorch data loaders ----------------------------------
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    testloader = DataLoader(
        test, #["train"],
        shuffle=True,
        batch_size=32,
        collate_fn=data_collator,
    )
    return testloader
    

def gen_evaluate_fn(
    model_name: str,
    testloader: DataLoader,
    device: torch.device,
):
    """Generate the function for centralized evaluation."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        net = get_model(model_name)
        set_params(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = test(net, testloader, device=device)
        return loss, {"centralized_f1": accuracy}

    return evaluate

# Define metric aggregation function
def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["f1"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    total_train_time = sum(m.get("train_time", 0.0) for _, m in metrics)
    
    # Aggregate and return custom metric (weighted average)
    return {
            "federated_evaluate_f1": sum(accuracies) / sum(examples),
            "avg_train_time": total_train_time / len(metrics)  # or use weighted avg if needed
        }
    
def server_fn(context: Context) -> ServerAppComponents:
    if hasattr(os, 'sched_setaffinity'):
        try:
            # Get available cores
            available_cores = os.sched_getaffinity(0)
            print(f"Available CPU cores: {available_cores}")
            
            # Choose highest core ID to avoid worker conflicts
            target_core = max(available_cores) if available_cores else 0
            
            os.sched_setaffinity(0, {target_core})
            print(f"✅ Server bound to CPU core {target_core} (PID: {os.getpid()})")
        except Exception as e:
            print(f"⚠️  Failed to set CPU affinity: {e}")
    else:
        print("❌ CPU affinity not supported on this platform")
    
    """Construct components for ServerApp."""

    num_rounds = context.run_config["num-server-rounds"]
    server_device = context.run_config["server-device"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    
    # Construct ServerConfig
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    # Set global model initialization
    model_name = context.run_config["model-name"]
    ndarrays = get_params(get_model(model_name))
    global_model_init = ndarrays_to_parameters(ndarrays)

    # global_test_set = load_dataset("ncbi/ncbi_disease")["test"]   
    # testloader = data_prep(model_name, global_test_set)
 
    # Define strategy
    strategy = CustomFedAvg(
        run_config=context.run_config,
        use_wandb=context.run_config["use-wandb"],
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        initial_parameters=global_model_init,
        # evaluate_fn=None,#gen_evaluate_fn(model_name, testloader, device=server_device),
        evaluate_metrics_aggregation_fn=weighted_average,
        )
    # strategy_ = FedAvg(
    #     fraction_fit=fraction_fit,
    #     fraction_evaluate=fraction_evaluate,
    #     initial_parameters=global_model_init,
    # )
    
    config = ServerConfig(num_rounds=num_rounds)
    # fl.server.start_server(server_address="localhost:8080", config={"num_rounds": num_rounds}, strategy=strategy)
    return ServerAppComponents(config=config, strategy=strategy)

app = ServerApp(server_fn=server_fn)
