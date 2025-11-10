# """huggingface_example: A Flower / Hugging Face app."""

# import warnings
# import torch, os, gc
# from flwr.client import Client, ClientApp, NumPyClient
# from flwr.common import Context
# from opacus import PrivacyEngine
# from opacus.noise_generator import NoiseGenerator
# # from transformers import (
# #     AutoTokenizer,
# #     # DataCollatorWithPadding,
# #     DataCollatorForTokenClassification,
# #     AutoModelForTokenClassification,
# # )
# # from transformers import logging
# from flwr.client.mod import LocalDpMod     # <- does clipping + Gaussian noise
# import logging
# from torch.optim import AdamW, Adafactor
# from huggingface_example.task import (
#     train,
#     test,
#     load_data,
#     set_params,
#     get_params,
#     get_model,
# )
# from torch.nn import DataParallel
# warnings.filterwarnings("ignore", category=FutureWarning)


# # class LaplaceNoise(NoiseGenerator):
# #     def __init__(self, noise_multiplier: float, max_grad_norm: float, device=None):
# #         super().__init__()
# #         self.noise_multiplier = noise_multiplier
# #         self.max_grad_norm = max_grad_norm
# #         self.device = device or torch.device("cpu")

# #     def __call__(self, sample_shape):
# #         # Scale similar to Gaussian mechanism but with Laplace distribution
# #         scale = self.noise_multiplier * self.max_grad_norm
# #         # Laplace(0, scale) = difference of two exponentials
# #         u = torch.rand(sample_shape, device=self.device) - 0.5
# #         noise = -scale * torch.sign(u) * torch.log1p(-2 * u.abs())
# #         return noise



# class LaplaceNoise(NoiseGenerator):
#     """
#     A simple Laplace noise generator compatible with Opacus' interface.

#     We model noise ~ Laplace(0, b) elementwise, where scale b is provided via
#     (noise_multiplier * max_grad_norm). In LDP usage with gradient clipping,
#     a common heuristic is noise_multiplier ≈ 1 / ε (assuming sensitivity ≈ max_grad_norm).
#     """

#     def __init__(self, noise_multiplier: float, max_grad_norm: float, device: Optional[torch.device] = None):
#         super().__init__()
#         self.noise_multiplier = float(noise_multiplier)
#         self.max_grad_norm = float(max_grad_norm)
#         self.device = device if device is not None else torch.device("cpu")

#     def __repr__(self) -> str:
#         return f"LaplaceNoise(scale={self.noise_multiplier * self.max_grad_norm:.6f})"

#     @torch.no_grad()
#     def sample(self, std: float, reference: torch.Tensor) -> torch.Tensor:
#         """
#         Opacus calls `sample(std, reference)` on the generator. We ignore `std`
#         (that's specific to Gaussian schedulers) and instead use our Laplace scale.
#         """
#         # Elementwise Laplace can be sampled by inverse-CDF:
#         # U ~ Uniform(-0.5, 0.5),  Laplace(0,b) = -b * sgn(U) * ln(1 - 2|U|)
#         u = torch.rand_like(reference, device=reference.device) - 0.5
#         b = self.noise_multiplier * self.max_grad_norm
#         # Avoid log(0) by clamping |u|
#         eps = 1e-12
#         x = -b * torch.sign(u) * torch.log(torch.clamp(1 - 2.0 * torch.abs(u), min=eps))
#         return x


# # Configure PyTorch memory settings
# torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for matmul
# torch.backends.cudnn.allow_tf32 = True        # Enable TF32 for convolutions
# torch.set_float32_matmul_precision('medium')  # Balance speed/accuracy

# # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# # os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3, 4, 5"
# # To mute warnings reminding that we need to train the model to a downstream task
# # This is something this example does.
# # logging.set_verbosity_error()

# # Keep each Ray actor to a single core
# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)

# # Disable the HF tokenizer’s own parallelism - it defeats the point above
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # CLIP_NORM = 1.0        # L2 clipping radius
# # EPSILON   = 8.0        # total privacy budget
# # DELTA     = 1e-5       # failure probability

# # # sensitivity is usually 1.0 if you clip *gradients*; keep the default
# # local_dp_obj = LocalDpMod(clipping_norm=CLIP_NORM,
# #                     sensitivity=1.0,
# #                     epsilon=EPSILON,
# #                     delta=DELTA)


# # Flower client
# class IMDBClient(NumPyClient):
#     def __init__(self, model_name, trainloader, testloader, local_epochs, 
#                  target_delta,
#                  noise_multiplier,
#                  max_grad_norm,) -> None:# Configure PyTorch memory settings
        
#         torch.set_num_threads(1)
#         # torch.set_num_interop_threads(1)
#         # torch.cuda.set_device(0)        # index 0 in _local_ CUDA_VISIBLE_DEVICES
        
#         self.device = torch.device("cuda") 
#         self.trainloader = trainloader
#         self.testloader = testloader
#         self.local_epochs = local_epochs
#         self.net = get_model(model_name)
#         # self.net = DataParallel(self.net)
#         self.net.to(self.device)
        
#         self.target_delta = target_delta
#         self.noise_multiplier = noise_multiplier
#         self.max_grad_norm = max_grad_norm

#     def fit(self, parameters, config) -> tuple[list, int, dict]:
#         torch.cuda.empty_cache()
#         set_params(self.net, parameters)
#         optimizer = AdamW(self.net.parameters(), lr=1e-5)
#         self.net.train()
        
#         # --- Privacy Engine with Laplace noise ---
#         privacy_engine = PrivacyEngine(secure_mode=False)
        
#         (
#             self.net,
#             optimizer,
#             self.trainloader,
#         ) = privacy_engine.make_private(
#             module = self.net,
#             optimizer = optimizer,
#             data_loader = self.trainloader,
#             noise_multiplier = self.noise_multiplier,
#             max_grad_norm = self.max_grad_norm,
#             noise_generator = LaplaceNoise(
#                 noise_multiplier=self.noise_multiplier,
#                 max_grad_norm=self.max_grad_norm,
#                 device=self.device,
#             ),
#         )
        
#         # --- Training loop ---
#         train_loss, train_time, epsilon = train(
#             self.net,
#             self.trainloader,
#             epochs=self.local_epochs,
#             device=self.device,
#             privacy_engine=privacy_engine,
#             optimizer=optimizer,
#             target_delta=self.target_delta,
#         )
        
#         if epsilon is not None:
#             print(f"Epsilon value for delta={self.target_delta} is {epsilon:.2f}")
#         else:
#             print("Epsilon value not available.") 
        
#         print(f"[Client] Training time: {train_time:.2f} seconds")
        
#         # --- Return parameters and metrics ---
#         result = (
#             get_params(self.net),
#             len(self.trainloader.dataset),
#             {
#                 "avg_train_loss": train_loss,
#                 "train_time": train_time,
#             },
#         )
        
#         # --- Clean up GPU/CPU memory ---
#         # self.net.to("cpu")
#         del optimizer, privacy_engine
#         torch.cuda.empty_cache()
#         gc.collect()    

#     def evaluate(self, parameters, config) -> tuple[float, int, dict[str, float]]:
#         set_params(self.net, parameters)
#         loss, f1 = test(self.net, self.testloader, device=self.device)
#         return float(loss), len(self.testloader.dataset), {"f1": float(f1)}

# def client_fn(context: Context) -> Client:
#     """Construct a Client that will be run in a ClientApp."""
#     # Read the node_config to fetch data partition associated to this node
#     partition_id = context.node_config["partition-id"]
#     num_partitions = context.node_config["num-partitions"]
#     local_epochs = context.run_config["local-epochs"]
#     # Read the run config to get settings to configure the Client
#     model_name = context.run_config["model-name"]
#     noise_multiplier = 1.0 if partition_id % 2 == 0 else 1.5
    
#     trainloader, testloader = load_data(partition_id, num_partitions, model_name, cache_dir = '../cache_dir')

#     return IMDBClient(model_name, trainloader, testloader, local_epochs, 
#                       context.run_config["target-delta"],
#                       noise_multiplier,
#                       context.run_config["max-grad-norm"]).to_client()

# app = ClientApp(client_fn=client_fn)

"""huggingface_example: A Flower / Hugging Face app."""

import warnings
import torch, os, gc
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
from opacus import PrivacyEngine
import logging
from torch.optim import AdamW
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


# -------------------------------
# Laplace noise helper
# -------------------------------
def add_laplace_noise(tensor, scale, device):
    dist = torch.distributions.Laplace(loc=0.0, scale=scale)
    noise = dist.sample(tensor.shape).to(device)
    return tensor + noise

# -------------------------------
# Flower client
# -------------------------------
class IMDBClient(NumPyClient):
    def __init__(self, model_name, trainloader, testloader, local_epochs,
                 target_delta, noise_multiplier, max_grad_norm,) -> None:
        
        torch.set_num_threads(1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.net = get_model(model_name).to(self.device)
        
        self.target_delta = target_delta
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm

    def fit(self, parameters, config) -> tuple[list, int, dict]:
        torch.cuda.empty_cache()
        set_params(self.net, parameters)
        optimizer = AdamW(self.net.parameters(), lr=1e-3)
        self.net.train()

        # -------------------------------
        # Training loop with Laplace DP
        # -------------------------------
        epoch_losses = []
        for epoch in range(self.local_epochs):
            running_loss = 0.0
            for batch in self.trainloader:
                optimizer.zero_grad()
                # Forward pass
                input_ids, attention_mask, labels = batch
                batch = {
                    "input_ids": input_ids.to(self.device),
                    "attention_mask": attention_mask.to(self.device),
                    "labels": labels.to(self.device),
                }
                outputs = self.net(**batch)
                loss = outputs.loss
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)

                # Add Laplace noise to gradients
                for p in self.net.parameters():
                    if p.grad is not None:
                        p.grad.data = add_laplace_noise(
                            p.grad.data,
                            scale=self.noise_multiplier * self.max_grad_norm,
                            device=self.device,
                        )

                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(self.trainloader)
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{self.local_epochs} - Loss: {avg_loss:.4f}")

        train_loss = float(sum(epoch_losses) / len(epoch_losses))
        train_time = 0.0  # placeholder if you want timing info

        print(f"[Client] Training complete. Avg loss: {train_loss:.4f}")

        # -------------------------------
        # Return parameters and metrics
        # -------------------------------
        result = (
            get_params(self.net),
            len(self.trainloader.dataset),
            {
                "avg_train_loss": train_loss,
                "train_time": train_time,
            },
        )

        del optimizer
        torch.cuda.empty_cache()
        gc.collect()
        return result

    def evaluate(self, parameters, config) -> tuple[float, int, dict[str, float]]:
        set_params(self.net, parameters)
        loss, f1 = test(self.net, self.testloader, device=self.device)
        return float(loss), len(self.testloader.dataset), {"f1": float(f1)}


# -------------------------------
# Client factory
# -------------------------------
def client_fn(context: Context) -> Client:
    """Construct a Client that will be run in a ClientApp."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    local_epochs = context.run_config["local-epochs"]
    model_name = context.run_config["model-name"]

    noise_multiplier = 0.05 if partition_id % 2 == 0 else 0.5
    trainloader, testloader = load_data(
        partition_id, num_partitions, model_name, cache_dir="../cache_dir"
    )

    return IMDBClient(
        model_name, trainloader, testloader, local_epochs,
        context.run_config["target-delta"],
        noise_multiplier,
        context.run_config["max-grad-norm"]
    ).to_client()


app = ClientApp(client_fn=client_fn)
