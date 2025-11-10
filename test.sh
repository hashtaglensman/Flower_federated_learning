#!/bin/bash
#SBATCH --job-name=flower_24c_3g_3n
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8  
#SBATCH --gres=gpu:3
#SBATCH --time=04:00:00
#SBATCH --mem=512G
#SBATCH --partition=batch
#SBATCH --exclusive
#SBATCH --output=flower_sim_%j.out
#SBATCH --error=flower_sim_%j.err
#SBATCH -A aupendrannair_anoop_meed_0001
#SBATCH --mail-user=aupendrannair@mail.smu.edu
#SBATCH -D /work/projects/aupendrannair/anoop_meed/anoop/work/Flower/quickstart-huggingface

# set -euo pipefail

# Load modules
module purge
module load conda gcc/11.2.0 cuda/11.8.0 cudnn/8.7.0.84
# module load conda gcc/11.2.0 cuda/12.4.1-nsrijq7 cudnn/9.3.0.75-12-fv273jo 

# Setup conda
CONDA_ROOT=/work/projects/aupendrannair/anoop_meed/anoop/miniconda3
conda activate /work/projects/aupendrannair/anoop_meed/anoop/envs/flwr

# Environment setup
export PROJECT_ROOT="/work/projects/aupendrannair/anoop_meed/anoop/work/Flower/quickstart-huggingface"
export VENV_PATH="/work/projects/aupendrannair/anoop_meed/anoop/envs/flwr"
# export PYTHONPATH="$PROJECT_ROOT:$VENV_PATH/lib/python3.11/site-packages:$PYTHONPATH"
export PATH="$VENV_PATH/bin:$PATH"
WORKDIR="/work/projects/aupendrannair/anoop_meed/anoop/work/Flower/quickstart-huggingface"

# Optional: make sure Ray/TensorFlow/NCCL see the single GPU allocated
# (SLURM usually sets CUDA_VISIBLE_DEVICES for you; keep this for clarity/logging)
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Clean shutdown of Ray when this script exits on any node
trap 'echo "[${SLURMD_NODENAME}] Stopping Ray..."; ray stop >/dev/null 2>&1 || true' EXIT

# ------------------------------------------------------------------
# Pick Ray head
# ------------------------------------------------------------------
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes[0]}
echo "Head Node: $head_node"
echo "All Nodes : ${nodes[@]}"

# ------------------------------------------------------------------
# Start Ray and run Flower
# ------------------------------------------------------------------
if [ "$SLURMD_NODENAME" = "$head_node" ]; then
    echo "[${SLURMD_NODENAME}] Starting Ray HEAD"
    # NOTE: No --address here on the head
    ray start --head --port=6379 --num-cpus=8 --num-gpus=1 &

    echo "[${SLURMD_NODENAME}] Waiting for Ray head to initialize..."
    # Wait until Ray head answers before launching Flower
    until ray status --address="${head_node}:6379" >/dev/null 2>&1; do
        sleep 2
    done
    echo "[${SLURMD_NODENAME}] Ray head is ready."

    echo "[${SLURMD_NODENAME}] Launching Flower (TOML: sim-24c-3g-3nodes)"
    # Flower connects with address="auto" from your TOML; do NOT pass num_cpus/num_gpus via init_args
    flwr run . sim-24c-3g-3nodes

    # After Flower finishes, Ray will be stopped by trap
else
    echo "[${SLURMD_NODENAME}] Starting Ray WORKER (connect to ${head_node}:6379)"
    # Worker connects to the head; advertise this node's resources
    ray start --address="${head_node}:6379" --num-cpus=8 --num-gpus=1 &

    # Keep the worker process alive for the duration of the job
    wait
fi
