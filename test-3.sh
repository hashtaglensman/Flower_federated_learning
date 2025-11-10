#!/bin/bash
#SBATCH --job-name=flower_16c_2g_2n
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8  
#SBATCH --gres=gpu:2
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

# ------------------------------------------------------------------
# Determine which node is the Ray head
# ------------------------------------------------------------------
# nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
# head_node=${nodes[0]}

# # ------------------------------------------------------------------
# # Start Ray and run Flower
# # ------------------------------------------------------------------
# if [ "$SLURMD_NODENAME" = "$head_node" ]; then
#     echo "Starting Ray HEAD on $head_node"
#     ray start --head --address="auto" --port=6379 --num-cpus=8 --num-gpus=1 &
#     sleep 10  # wait for head to initialize

#     echo "Running Flower simulation with TOML config"
#     flwr run . sim-16c-2g-2nodes
# else
#     echo "Starting Ray WORKER on $SLURMD_NODENAME"
#     ray start --address="${head_node}:6379" --num-cpus=8 --num-gpus=1 &
#     wait  # keep worker process alive
# fi

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
head_node=${nodes[0]}

if [ "$SLURMD_NODENAME" = "$head_node" ]; then
    echo "Starting Ray HEAD on $head_node"
    # ✅ No --address here
    ray start --head --port=6379 --num-cpus=8 --num-gpus=1 &

    # Wait for Ray head to become reachable
    echo "Waiting for Ray head to initialize..."
    until ray status --address="${head_node}:6379" >/dev/null 2>&1; do
        sleep 2
    done
    echo "Ray head is ready. Launching Flower"

    # Flower will internally use address="auto" from the TOML
    flwr run . sim-16c-2g-2nodes

else
    echo "Starting Ray WORKER on $SLURMD_NODENAME"
    # ✅ Worker connects to the head
    ray start --address="${head_node}:6379" --num-cpus=8 --num-gpus=1 &
    wait
fi
