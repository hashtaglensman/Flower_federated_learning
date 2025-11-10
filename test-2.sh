#!/bin/bash
#SBATCH --job-name=flower_64c_8g_1n
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64   
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --mem=512G
#SBATCH --partition=batch
#SBATCH --exclusive
#SBATCH --output=flower_sim_%j.out
#SBATCH --error=flower_sim_%j.err
#SBATCH -A aupendrannair_anoop_meed_0001
#SBATCH --mail-user=aupendrannair@mail.smu.edu
#SBATCH -D /work/projects/aupendrannair/anoop_meed/anoop/work/Flower/quickstart-huggingface

# --- Modules ---
module purge
module load conda gcc/11.2.0 cuda/11.8.0 cudnn/8.7.0.84
# module load conda gcc/11.2.0 cuda/12.4.1-nsrijq7 cudnn/9.3.0.75-12-fv273jo

# --- Conda env ---
CONDA_ROOT=/work/projects/aupendrannair/anoop_meed/anoop/miniconda3
conda activate /work/projects/aupendrannair/anoop_meed/anoop/envs/flwr

# --- Paths ---
export PROJECT_ROOT="/work/projects/aupendrannair/anoop_meed/anoop/work/Flower/quickstart-huggingface"
export VENV_PATH="/work/projects/aupendrannair/anoop_meed/anoop/envs/flwr"
export PATH="$VENV_PATH/bin:$PATH"
WORKDIR="/work/projects/aupendrannair/anoop_meed/anoop/work/Flower/quickstart-huggingface"
cd "$WORKDIR"

# --- Safety: stop Ray on exit ---
trap 'ray stop >/dev/null 2>&1 || true' EXIT

echo "Starting local Ray HEAD on single node"
# Start a local Ray head only; CPUs match SLURM allocation, GPUs=2 on this node
ray start --head --port=6379 --num-cpus="${SLURM_CPUS_PER_TASK:-16}" --num-gpus=2

echo "Waiting for Ray to become ready..."
until ray status --address="127.0.0.1:6379" >/dev/null 2>&1; do
  sleep 2
done
echo "Ray is ready."

# Let Flower auto-discover the local Ray
export RAY_ADDRESS=auto

# --- Run Flower ---
# Use your single-node sim config name; change if your TOML uses a different key
flwr run . sim-64c-8g

# Ray will be stopped by the trap on exit
