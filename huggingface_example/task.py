"""huggingface_example: A Flower / Hugging Face app."""

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Initialize TF first without GPU
from transformers import BertForTokenClassification, AutoConfig
from torch.cuda.amp import autocast, GradScaler
from opacus.utils.batch_memory_manager import BatchMemoryManager

# import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')
from typing import Any, Tuple, List, Sequence
from collections import OrderedDict
from datasets import load_dataset
import torch, random
from time import time  # at the top
from evaluate import load  #as load_metric
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    # DataCollatorWithPadding,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
)
from datasets.utils.logging import disable_progress_bar
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner,  DirichletPartitioner
from flwr.common.typing import UserConfig
import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from collections.abc import Iterable
# from sklearn.metrics import f1_score, classification_report
import numpy as np
from seqeval.metrics import classification_report
from tqdm import tqdm
disable_progress_bar()
from tqdm import tqdm


fds = None  # Cache FederatedDataset
# scaler = GradScaler()

def load_data(
    partition_id: int,
    num_partitions: int,
    model_name: str,
    cache_dir: str,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Load NCBI data (random 1,000 samples per client: 800 train, 200 eval)"""
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="ncbi/ncbi_disease",
            partitioners={"train": partitioner},
            trust_remote_code=True
        )

    # Load this client's partition
    partition = fds.load_partition(partition_id)

    # Randomly shuffle and select 1000 samples from the client's partition
    partition = partition.shuffle(seed=42) #.select(range(1000))  # <- core fix
    # ➋ Take exactly 1 000 examples
    desired = 1000
    rng = random.Random(42)             # deterministic for repeatability
    full_idx = list(range(len(partition)))

    if len(full_idx) >= desired:        # enough data → sample w/out replacement
        chosen = rng.sample(full_idx, desired)
    else:                               # too little → sample WITH replacement
        chosen = rng.choices(full_idx, k=desired)

    partition = partition.select(chosen)

    # Split: 800 train, 200 eval
    split = partition.train_test_split(test_size=200, seed=42)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=512,
        cache_dir=cache_dir,
        # force_download=True,
    )

    def tuple_collator(batch):
        collator = DataCollatorForTokenClassification(tokenizer)
        batch_dict = collator(batch)
        return (
            batch_dict["input_ids"],
            batch_dict["attention_mask"],
            batch_dict["labels"]
        )

    def tokenize_and_align_labels(batch):
        tokenized = tokenizer(
            batch["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding = "max_length",
            max_length = 512,
        )

        aligned_labels = []
        for i in range(len(batch["tokens"])):
            word_ids = tokenized.word_ids(batch_index=i)
            label_ids = []
            prev_word = None

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != prev_word:
                    label_ids.append(batch["ner_tags"][i][word_idx])
                else:
                    label_ids.append(batch["ner_tags"][i][word_idx])
                prev_word = word_idx

            aligned_labels.append(label_ids)

        tokenized["labels"] = aligned_labels
        return tokenized

    # Tokenize both splits
    split = split.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=["tokens", "ner_tags", "id"]
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainloader = DataLoader(
        split["train"],
        shuffle=True,
        batch_size=8,
        collate_fn=tuple_collator,#data_collator,
        drop_last=True,
    )

    valloader = DataLoader(
        split["test"],
        batch_size=8,
        collate_fn=data_collator,
        # drop_last=True,
    )

    return trainloader, valloader

# def get_model_(model_name):
#     label2id = {"O": 0, "B-Disease": 1, "I-Disease": 2}
#     id2label = {0: "O", 1: "B-Disease", 2: "I-Disease"}
    
#     # Create config with correct number of labels
#     config = AutoConfig.from_pretrained(
#         model_name,
#         num_labels=3,
#         id2label=id2label,
#         label2id=label2id
#     )
    
#     # Initialize model with random weights
#     model = AutoModelForTokenClassification.from_config(config)
    
#     # Load base BERT weights (without classifier)
#     base_model = AutoModel.from_pretrained(
#         model_name,
#         cache_dir='../cache_dir',
#         add_pooling_layer=False  # Important for matching architecture
#     )
#     model.base_model.load_state_dict(base_model.state_dict(), strict=True)
    
#     del base_model  # Free memory
#     return model

def get_model(model_name: str):
    label2id = {"O": 0, "B-Disease": 1, "I-Disease": 2}
    id2label = {0: "O", 1: "B-Disease", 2: "I-Disease"}
    # return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    return AutoModelForTokenClassification.from_pretrained(model_name, num_labels=3, output_attentions=False,  ignore_mismatched_sizes=True, cache_dir = '../cache_dir', output_hidden_states=False, id2label=id2label, label2id=label2id)  


# def get_model(model_name: str, cache_dir: str = "../cache_dir"):
#     label2id = {"O": 0, "B-Disease": 1, "I-Disease": 2}
#     id2label = {v: k for k, v in label2id.items()}

#     # 1️⃣ build a fresh token-classification model skeleton
#     cfg = AutoConfig.from_pretrained(
#         model_name,
#         num_labels=len(label2id),
#         id2label=id2label,
#         label2id=label2id,
#     )
#     model = BertForTokenClassification(cfg)

#     # 2️⃣ load *only* the base encoder weights
#     base = AutoModel.from_pretrained(model_name, cache_dir=cache_dir,
#                                      add_pooling_layer=False)
#     model.base_model.load_state_dict(base.state_dict(), strict=True)

#     # classifier layer now has random `[3, 768]` weights → fine-tune as usual
#     return model
    

# def get_model(model_name):
#     label2id = {"O": 0, "B-Disease": 1, "I-Disease": 2}
#     id2label = {0: "O", 1: "B-Disease", 2: "I-Disease"}
#     # return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

#     # 1. Build a model skeleton with new label set
#     config = AutoConfig.from_pretrained(checkpoint_path, num_labels=len(label2id),
#                                         id2label=id2label, label2id=label2id)
#     model = BertForTokenClassification(config)

#     # 2. Load the old weights but **ignore** the classifier
#     state_dict = torch.load(f"{checkpoint_path}/pytorch_model.bin", map_location="cpu")
#     filtered = {k: v for k, v in state_dict.items() if not k.startswith("classifier.")}
#     missing, unexpected = model.load_state_dict(filtered, strict=False)
#     # missing → new head params, expected; unexpected → old head we skipped

#     # 3. Now fine-tune as usual
#     return model

    

    
    # return AutoModelForTokenClassification.from_pretrained(model_name, num_labels=3, output_attentions=False, cache_dir = '../cache_dir', output_hidden_states=False, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)      
    
def get_params(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_params(model, parameters) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def set_params_(model: torch.nn.Module, parameters: Sequence) -> None:
    """
    Copy a list/tuple of numpy- or torch-arrays into `model.state_dict()`.

    * Tensors whose **shape matches** the target tensor are copied verbatim.
    * Tensors whose **shape differs** are **skipped** (you’ll see a warning).
      The layer keeps whatever values it already had (usually random init).

    This prevents “size mismatch for classifier.*” when, for example,
    you switch from a 768-label MLM head to a 3-label token-classification
    head in federated training.
    """
    # 1️⃣ current model parameters
    state_dict = model.state_dict()

    # 2️⃣ pair incoming tensors with keys and load selectively
    new_state = OrderedDict()
    for (name, old_tensor), new_tensor in zip(state_dict.items(), parameters):
        new_tensor = torch.as_tensor(new_tensor)

        if old_tensor.shape == new_tensor.shape:
            new_state[name] = new_tensor
        else:
            # keep the existing tensor (leave it unchanged in the state-dict)
            new_state[name] = old_tensor
            print(
                f"[set_params] ‼️  skipped '{name}': "
                f"expected {tuple(old_tensor.shape)}, got {tuple(new_tensor.shape)}"
            )

    # 3️⃣ load, allowing “missing/unexpected” (we already filtered sizes)
    model.load_state_dict(new_state, strict=True)

def randomized_response_on_deltas(
    deltas, 
    *, 
    flip_prob: float | None = None, 
    epsilon: float | None = None, 
    device: torch.device | None = None
):
    """
    Apply Randomized Response (RR) to the *signs* of parameter deltas.

    Args:
        deltas: list/sequence of numpy or torch tensors (parameter differences)
        flip_prob: probability of flipping the sign bit, in [0,1]
        epsilon: optional ε; if provided, flip_prob = 1 - e^ε / (e^ε + 1)
        device: CUDA/CPU device for RNG; defaults to CPU

    Returns:
        List of numpy arrays with RR-perturbed deltas.
    """
    if epsilon is not None:
        # Classic RR mapping for binary responses:
        # keep_prob = e^ε / (e^ε + 1)  → flip_prob = 1 - keep_prob
        keep_prob = float(np.exp(epsilon) / (np.exp(epsilon) + 1.0))
        flip_prob = 1.0 - keep_prob

    if flip_prob is None:
        flip_prob = 0.2  # sensible default if nothing provided

    flip_prob = float(max(0.0, min(1.0, flip_prob)))
    device = device or torch.device("cpu")

    out = []
    for arr in deltas:
        t = torch.as_tensor(arr, device=device, dtype=torch.float32)
        if t.numel() == 0:
            out.append(t.detach().cpu().numpy())
            continue

        # sign ∈ {-1, +1}, magnitude ≥ 0
        sign = torch.sign(t)
        mag = t.abs()

        # Bernoulli mask: True means "flip this sign"
        flips = torch.rand_like(t).lt(flip_prob)
        # multiply by -1 where flips==True  →  (1 - 2*flip)
        sign_pert = sign * torch.where(flips, torch.tensor(-1.0, device=device), torch.tensor(1.0, device=device))

        t_pert = sign_pert * mag
        out.append(t_pert.detach().cpu().numpy())

    return out


def train_wo_dp(net, trainloader, epochs, device, optimizer=None, grad_clip_norm=None):
    """
    Standard (non-DP) training loop.

    Args:
        net: torch.nn.Module
        trainloader: DataLoader yielding (input_ids, attention_mask, labels) or dict
        epochs: int
        device: torch.device
        optimizer: torch.optim.Optimizer (if None, AdamW(1e-5) is created)
        grad_clip_norm: float or None, to enable gradient clipping

    Returns:
        epoch_losses: list[float]  # average loss per epoch
        total_time: float          # seconds
    """
    net.to(device)
    net.train()
    epoch_losses = []
    start_time = time()

    # Perf hints (safe without DP)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Fallback optimizer if not provided
    if optimizer is None:
        from torch.optim import AdamW
        optimizer = AdamW(net.parameters(), lr=1e-5)

    for epoch in range(epochs):
        running_loss = 0.0

        for batch in tqdm(trainloader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad(set_to_none=True)

            # Normalize batch format
            if isinstance(batch, tuple):
                input_ids, attention_mask, labels = batch
                batch = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # Forward / backward / step
            outputs = net(**batch)
            loss = outputs.loss
            loss.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip_norm)

            optimizer.step()

            running_loss += float(loss.item())

            # Free transient tensors promptly
            del loss, outputs
            if device.type == "cuda":
                torch.cuda.empty_cache()

        avg_epoch_loss = running_loss / len(trainloader)
        epoch_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_epoch_loss:.4f}")

    total_time = time() - start_time
    return epoch_losses, total_time
       

def train__(
    net: torch.nn.Module,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,
    privacy_engine,
    optimizer: torch.optim.Optimizer,
    target_delta: float,
    ):
    """Train with gradient accumulation (no BatchMemoryManager).

    Parameters
    ----------
    net : torch.nn.Module
        The model to train.
    trainloader : DataLoader
        Loader returning *dict* batches (HF token-classification).
    epochs : int
        Number of passes over the loader.
    device : torch.device
        GPU / CPU device.
    privacy_engine : opacus.PrivacyEngine
        Used only to query epsilon at the end.
    optimizer : torch.optim.Optimizer
        Optimizer already wrapped by ``privacy_engine.make_private``.
    target_delta : float
        δ for (ε,δ)-DP accounting.
    accumulation_steps : int, optional
        Number of steps to accumulate before ``optimizer.step()``.
    """

    net.to(device)
    net.train()

    # --- mixed-precision helpers -------------------------------------------------
    # scaler = GradScaler(enabled=True)

    # --- speed tweaks -----------------------------------------------------------
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    epoch_losses: list[float] = []
    start_wall = time()

    accumulation_steps = 4
    
    for epoch in range(epochs):
        running_loss = 0.0
       
        for step, batch in enumerate(tqdm(trainloader, desc=f" Training Epoch {epoch+1}")):
            optimizer.zero_grad(set_to_none=True)
            if isinstance(batch, tuple):
                input_ids, attention_mask, labels = batch
                batch = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                        }
            batch = {k: v.to(device) for k, v in batch.items()}

            # with autocast():
            outputs = net(**batch)
            loss = outputs.loss  # scale down

            loss = loss / accumulation_steps

            # Backward pass
            loss.backward()
            
            # Take an optimizer step every `accumulation_steps`
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()    # Opacus optimizer will clip, noise, and average gradients
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * accumulation_steps  # undo division for logging

        avg_epoch_loss = running_loss / len(trainloader)
        epoch_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{epochs} – Loss: {avg_epoch_loss:.4f}")

    epsilon = privacy_engine.get_epsilon(delta=target_delta)
    total_time = time() - start_wall
    return epoch_losses, total_time, epsilon


def train(net, trainloader, epochs, device, privacy_engine, optimizer, target_delta):
    net.to(device)
    net.train()
    epoch_losses = []
    epsilons = []   # track epsilon per epoch
    start_time = time()
    
    # Memory optimization flags
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Use BatchMemoryManager to handle physical batches
    max_physical_batch_size = 2  # Adjust based on your GPU capacity
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        with BatchMemoryManager(
            data_loader=trainloader,
            max_physical_batch_size=max_physical_batch_size,
            optimizer=optimizer
        ) as memory_safe_loader:
            for batch in tqdm(memory_safe_loader, desc=f"Training Epoch {epoch+1}"):
                optimizer.zero_grad(set_to_none=True)
                
                if isinstance(batch, tuple):
                    input_ids, attention_mask, labels = batch
                    batch = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                    }
                    
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                outputs = net(**batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                del loss, outputs
                torch.cuda.empty_cache()
                
        # Calculate epoch loss
        avg_epoch_loss = running_loss / len(trainloader)
        epoch_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_epoch_loss:.4f}")
        
        # --- calculate epsilon after each epoch ---
        try:
            epsilon = privacy_engine.get_epsilon(delta=target_delta)
            if epsilon is None or not np.isfinite(epsilon):
                epsilon = 0.0
        except Exception as e:
            print(f"⚠️ Error calculating epsilon at epoch {epoch+1}: {e}")
            epsilon = 0.0
        
        epsilons.append(epsilon)
        print(f"Epoch {epoch + 1} - ε (delta={target_delta}): {epsilon:.2f}")
    
    total_time = time() - start_time
    
    # Return per-epoch losses + epsilons + total_time
    return epoch_losses, total_time, epsilons
 
    
# def test(net, testloader, device) -> tuple[Any | float, Any]:
#     metric = load_metric("accuracy")
#     loss = 0
#     net.eval()
#     for batch in testloader:
#         batch = {k: v.to(device) for k, v in batch.items()}
#         with torch.no_grad():
#             outputs = net(**batch)
#         logits = outputs.logits
#         loss += outputs.loss.item()
#         predictions = torch.argmax(logits, dim=-1)
#         metric.add_batch(predictions=predictions, references=batch["labels"])
#     loss /= len(testloader.dataset)
#     accuracy = metric.compute()["accuracy"]
#     return loss, accuracy


def test_(net, testloader, device) -> tuple[float, float]:
    """
    Evaluate on the validation set.

    Returns
    -------
    loss  : float   – mean cross-entropy per batch
    f1    : float   – token-level (micro) F1 from `seqeval`
    """
    net.to(device)
    metric = load("seqeval")
    eval_loss = 0.0
    id2label = net.config.id2label

    net.eval()
    for batch in testloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = net(**batch)

        # accumulate loss
        eval_loss += outputs.loss.item()

        # logits → predicted IDs
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        labels = batch["labels"].cpu().numpy()

        # convert to lists of label strings, ignoring -100
        true_preds, true_labels = [], []
        for p_seq, l_seq in zip(preds, labels):
            pred_tags, gold_tags = [], []
            for p, l in zip(p_seq, l_seq):
                if l != -100:                        # skip special/pad tokens
                    pred_tags.append(id2label[p])
                    gold_tags.append(id2label[l])
            true_preds.append(pred_tags)
            true_labels.append(gold_tags)

        metric.add_batch(predictions=true_preds, references=true_labels)

    eval_loss /= len(testloader)
    f1 = metric.compute()["overall_f1"]
    return eval_loss, f1

def test(net, testloader, device) -> tuple[float, float]:
    """
    Evaluate on the validation set.

    Returns
    -------
    loss  : float   – mean cross-entropy per batch
    f1    : float   – token-level (micro) F1 from `seqeval`
    """
    net.to(device)
    eval_loss = 0.0
    id2label = net.config.id2label

    # Initialize containers for full predictions and labels
    all_true_predictions = []
    all_true_labels = []

    net.eval()
    for batch in tqdm(testloader, "Testing"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = net(**batch)

        # Accumulate loss
        eval_loss += outputs.loss.item()

        # Get predictions and labels
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        labels = batch["labels"].cpu().numpy()

        # Convert to label strings while filtering padding
        batch_preds = []
        batch_labels = []
        for i in range(len(preds)):
            seq_preds = []
            seq_labels = []
            for j in range(len(preds[i])):
                if labels[i][j] != -100:  # Skip padding tokens
                    seq_preds.append(id2label[preds[i][j]])
                    seq_labels.append(id2label[labels[i][j]])
            batch_preds.append(seq_preds)
            batch_labels.append(seq_labels)
        
        # Add to global containers
        all_true_predictions.extend(batch_preds)
        all_true_labels.extend(batch_labels)

    # Calculate average loss
    eval_loss /= len(testloader)
    
    # Compute final metrics
    report = classification_report(
        all_true_labels,
        all_true_predictions,
        output_dict=True,
        zero_division=0
    )
    # Extract overall F1 score
    f1 = report["micro avg"]["f1-score"]
    return eval_loss, f1


# def test(
#     net,
#     testloader,
#     device: torch.device
# ) -> tuple[float, float]:
#     """
#     Evaluate on `testloader` without `seqeval`.

#     Returns
#     -------
#     loss : float   – mean cross-entropy per mini-batch
#     f1   : float   – token-level micro-F1
#     """
#     net.to(device)
#     id2label = net.config.id2label

#     net.eval()
#     cum_loss, n_batches = 0.0, 0
#     all_preds, all_labels = [], []

#     with torch.no_grad():
#         for batch in testloader:
#             batch = {k: v.to(device) for k, v in batch.items()}
#             outputs = net(**batch)

#             # accumulate loss
#             cum_loss += outputs.loss.item()
#             n_batches += 1

#             # logits → predicted IDs
#             preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
#             labels = batch["labels"].cpu().numpy()

#             # collect flattened predictions / gold labels
#             fp, fl = _flatten(preds, labels, id2label, ignore_idx=-100)
#             all_preds.extend(fp)
#             all_labels.extend(fl)

#     # compute micro-averaged F1
#     f1 = f1_score(all_labels, all_preds, average="micro")
#     # (optional) print a full report
#     print(classification_report(all_labels, all_preds, digits=3))

#     return cum_loss / n_batches, f1

def create_run_dir(config: UserConfig) -> Path:
    """Create a directory where to save results from this run."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    # Save path is based on the current directory
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=True)

    # Save run config as json
    with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp)

    return save_path, run_dir