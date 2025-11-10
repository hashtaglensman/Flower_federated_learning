---
tags: [federated-learning, privacy-preservation, differential-privacy, huggingface, transformers, nlp, ncbi-disease, bert]
dataset: [NCBI-Disease]
framework: [transformers, flower, opacus]
---

# Privacy-Preserving Federated Learning with HuggingFace Transformers and Flower

This repository demonstrates advanced privacy-preserving federated learning implementations using HuggingFace Transformers, Flower, and various differential privacy techniques. The project focuses on Named Entity Recognition (NER) for disease identification using the NCBI-Disease dataset with BERT-based models.

## Overview

This comprehensive example showcases multiple privacy preservation techniques in federated learning:

- **Differential Privacy (DP)** with Opacus integration
- **Local Differential Privacy (LDP)** with Laplace mechanism
- **Personalized LDP** with per-client privacy budgets
- **Randomized Response** for parameter perturbation
- **Secure Aggregation** with SecAgg+ protocol
- **Non-private baseline** for performance comparison

## Key Features

### Privacy-Preserving Techniques

1. **Differential Privacy with Opacus**: Integration with Opacus library for gradient-level privacy protection
2. **Laplace Mechanism**: Custom Laplace noise generator for L1 sensitivity bounds
3. **Personalized Privacy Budgets**: Per-client configurable epsilon values (ε = 2.0, 6.0, 10.0)
4. **Randomized Response**: Binary perturbation of parameter updates
5. **Secure Aggregation**: SecAgg+ protocol for secure parameter aggregation
6. **Gradient Clipping**: L2 norm clipping with configurable thresholds

### Technical Implementation

- **Model**: BERT-base-uncased for token classification (3 labels: O, B-Disease, I-Disease)
- **Dataset**: NCBI-Disease dataset with 1000 samples per client (800 train, 200 eval)
- **Task**: Named Entity Recognition for disease identification
- **Framework**: Flower for federated learning orchestration
- **Memory Management**: GPU memory optimization with proper cleanup
- **CPU Affinity**: Server-side CPU core binding for performance optimization

## Project Structure

```
quickstart-huggingface/
├── huggingface_example/
│   ├── __init__.py
│   ├── client_app.py              # Main DP client with Opacus
│   ├── client_app_no_dp.py        # Non-private baseline client
│   ├── client_app_w_dp.py         # Differential privacy client
│   ├── client_app_pldp.py         # Personalized LDP with Laplace
│   ├── client_app_rr.py           # Randomized response client
│   ├── client_app_seg.py          # Secure aggregation client
│   ├── server_app.py              # Main server application
│   ├── server_app_seg.py          # Server with SecAgg+ support
│   ├── strategy.py                # Custom FedAvg with logging
│   ├── task.py                    # Model, training, and data utilities
│   ├── workflow_with_log.py       # SecAgg+ workflow with logging
│   └── test.ipynb                 # Jupyter notebook for testing
├── pyproject.toml                 # Project configuration and dependencies
├── server_runtimes.csv           # Server performance logs
├── slurm/                        # SLURM cluster configurations
└── README.md                     # This file
```

## Privacy Configurations

### 1. No Privacy (Baseline)
- **File**: `client_app_no_dp.py`
- **Description**: Standard federated learning without privacy protection
- **Use Case**: Performance baseline comparison

### 2. Differential Privacy with Opacus
- **File**: `client_app_w_dp.py`
- **Parameters**: 
  - Clipping norm: 1.0
  - Epsilon: 8.0
  - Delta: 1e-5
- **Mechanism**: Gaussian noise with privacy accounting

### 3. Personalized Local Differential Privacy
- **File**: `client_app_pldp.py`
- **Features**:
  - Per-client privacy budgets (ε = 2.0, 6.0, 10.0)
  - Laplace mechanism with L1 sensitivity
  - Global gradient clipping
  - Configurable via `personalized-epsilon-map`

### 4. Randomized Response
- **File**: `client_app_rr.py`
- **Mechanism**: Binary perturbation of parameter updates
- **Parameters**: Configurable flip probability and epsilon

### 5. Secure Aggregation
- **File**: `client_app_seg.py`
- **Protocol**: SecAgg+ for secure parameter aggregation
- **Features**: Dropout simulation and secure computation

## Installation

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/hashtaglensman/Flower_federated_learning.git
cd quickstart-huggingface

# Install dependencies
pip install -e .
```

### BERT Model Weights Download

The BERT-base-uncased model weights need to be downloaded and cached before running experiments. The model will be automatically downloaded on first use, but you can pre-download it to avoid delays:

```bash
# Pre-download BERT model weights (optional but recommended)
python -c "from transformers import AutoModelForTokenClassification, AutoTokenizer; AutoModelForTokenClassification.from_pretrained('bert-base-uncased', num_labels=3); AutoTokenizer.from_pretrained('bert-base-uncased')"
```

The model weights will be stored in the default HuggingFace cache directory (`~/.cache/huggingface/transformers/`). You can customize the cache location by setting the `HF_HOME` or `TRANSFORMERS_CACHE` environment variable:

```bash
# Use custom cache directory
export HF_HOME=/path/to/cache_dir
# or
export TRANSFORMERS_CACHE=/path/to/cache_dir
```

**Note**: Ensure sufficient disk space (~440MB for BERT-base-uncased model) and stable internet connection for the initial download.

### Dependencies

Key dependencies (see `pyproject.toml` for complete list):
- `flwr[simulation]>=1.19.0` - Flower federated learning framework
- `torch==2.6.0` - PyTorch deep learning framework
- `transformers>=4.30.0` - HuggingFace transformers
- `opacus` - Differential privacy library
- `wandb==0.17.8` - Weights & Biases logging
- `seqeval>=1.2.2` - Sequence evaluation metrics

## Configuration

### Privacy Parameters

Configure privacy settings in `pyproject.toml`:

```toml
[tool.flwr.app.config]
# General settings
num-server-rounds = 10
model-name = "bert-base-uncased"
local-epochs = 1

# Privacy parameters
target-delta = 0.01
max-grad-norm = 1.0

# Randomized response parameters
rr-flip-prob = 0.2
rr-epsilon = 1.0

# Personalized privacy budgets
[tool.flwr.app.config.personalized-epsilon-map]
"0" = 2.0   # High privacy (most sensitive)
"1" = 6.0   # Medium privacy
"2" = 10.0  # Low privacy (least sensitive)
```

### Resource Allocation

Configure GPU resources for different scales:

```toml
[tool.flwr.federations]
# 8 clients, 1 GPU
sim-8c-1g.num-supernodes = 8
sim-8c-1g.backend.client-resources.num-gpus = 0.125

# 16 clients, 2 GPUs
sim-16c-2g.num-supernodes = 16
sim-16c-2g.backend.client-resources.num-gpus = 0.125

# Up to 64 clients, 8 GPUs
sim-64c-8g.num-supernodes = 64
sim-64c-8g.backend.client-resources.num-gpus = 0.125
```

## Running Experiments

### Basic Simulation

```bash
# Run with default configuration (CPU only)
flwr run .
```

### GPU-Accelerated Simulation

```bash
# Run with GPU support
flwr run . local-simulation-gpu

# Run with specific client scale
flwr run . sim-16c-2g
flwr run . sim-32c-4g
flwr run . sim-64c-8g
```

### Custom Configuration

```bash
# Override configuration parameters
flwr run --run-config "num-server-rounds=5 local-epochs=2"

# Custom privacy settings
flwr run --run-config "target-delta=0.001 max-grad-norm=0.5"
```

### Privacy-Specific Runs

```bash
# Run with personalized LDP
flwr run . --run-config "clientapp=huggingface_example.client_app_pldp:app"

# Run with randomized response
flwr run . --run-config "clientapp=huggingface_example.client_app_rr:app"

# Run with secure aggregation
flwr run . --run-config "serverapp=huggingface_example.server_app_seg:app clientapp=huggingface_example.client_app_seg:app"
```

## Performance Monitoring

### Weights & Biases Integration

The project includes comprehensive logging with W&B:
- Training loss and F1 scores
- Privacy metrics (epsilon, delta)
- Training time per client
- Server runtime statistics
- Model checkpointing on best F1 score

### Metrics Tracked

- **Federated Evaluation F1**: Weighted average F1 score across clients
- **Centralized F1**: Server-side evaluation on held-out test set
- **Training Time**: Per-client training duration
- **Privacy Budget**: Epsilon consumption tracking
- **Memory Usage**: GPU memory utilization

## Advanced Features

### CPU Affinity Management

Server automatically binds to highest available CPU core to avoid conflicts with worker processes:

```python
# Automatic CPU affinity setting
if hasattr(os, 'sched_setaffinity'):
    target_core = max(available_cores)
    os.sched_setaffinity(0, {target_core})
```

### Memory Optimization

- GPU memory cleanup after each training round
- Gradient accumulation with memory management
- Batch memory management for large models

### Custom Noise Generators

Implementation of Laplace noise generator for L1 sensitivity:

```python
class LaplaceNoise(NoiseGenerator):
    def sample(self, std: float, reference: torch.Tensor) -> torch.Tensor:
        # Element-wise Laplace sampling
        u = torch.rand_like(reference) - 0.5
        b = self.noise_multiplier * self.max_grad_norm
        noise = -b * torch.sign(u) * torch.log(torch.clamp(1 - 2.0 * torch.abs(u), min=eps))
        return noise
```

## SLURM Cluster Support

The repository includes SLURM configuration files for high-performance computing clusters:

```bash
# Interactive session
sbatch slurm/01_interactive_session

# Array jobs for multiple experiments
sbatch slurm/06_sbatch_array.sbatch
```

## Research Applications

This implementation supports research in:
- **Privacy-Utility Trade-offs**: Compare different privacy mechanisms
- **Personalized Privacy**: Study impact of heterogeneous privacy budgets
- **Scalability Analysis**: Evaluate performance across different scales
- **Mechanism Comparison**: Benchmark DP, LDP, RR, and SecAgg+

## Citation

If you use this code in your research, please cite:

```bibtex
@software{flower_huggingface_privacy,
  title = {Privacy-Preserving Federated Learning with HuggingFace and Flower},
  author = {Krishnan, Anoop and Flower Authors},
  year = {2024},
  url = {https://github.com/hashtaglensman/Flower_federated_learning.git}
}
```

## Contributing

Contributions are welcome! Areas for improvement:
- Additional privacy mechanisms
- Performance optimizations
- New datasets and tasks
- Advanced aggregation strategies
- Privacy accounting improvements

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Support

For questions and support:
- Open an issue on GitHub
- Check Flower documentation: https://flower.ai/docs/
- Review HuggingFace documentation: https://huggingface.co/docs

---

**Note**: This is an advanced research implementation demonstrating various privacy-preserving techniques in federated learning. Ensure proper understanding of differential privacy concepts before deploying in production environments.
 
