#!/bin/bash
#SBATCH --job-name=gpt2
#SBATCH --nodes=1
#SBATCH --gres=gpu:4                 # 4 GPUs no mesmo nó
#SBATCH --ntasks-per-node=4          # 1 processo por GPU
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=vbertalan@gmail.com
#SBATCH --mail-type=ALL
# (opcional) #SBATCH --account=def-aloise

echo "Training GPT2..."
echo "Starting multi-GPU training job on $(hostname)"
echo "GPUs available on this node:"
nvidia-smi

# ====== MÓDULOS ======
module load StdEnv/2023
module load scipy-stack arrow
# Se seu PyTorch pip já inclui CUDA runtime, NÃO carregue 'cuda' aqui.

# ====== AMBIENTE VENV ======
source /home/vberta/projects/def-aloise/vberta/vbertapy/bin/activate

# ====== CACHES LOCAIS ======
export HF_HOME="$SLURM_TMPDIR/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PIP_CACHE_DIR="$SLURM_TMPDIR/.cache/pip"
mkdir -p "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR"

# ====== THREADS ======
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
ulimit -n 4096

# ====== NCCL/DDP – single-node estável ======
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29400

# Força uso do NCCL (evita backend MPI)
export ACCELERATE_USE_NCCL=1
export PYTORCH_DISTRIBUTED_BACKEND=nccl

# Configs básicas do NCCL
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_TIMEOUT=7200

# Evitar fragmentação do CUDA allocator
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

echo "PWD: $(pwd)"
cd /home/vberta/projects/def-aloise/vberta/Paper3/
ls -l gpt2.py || { echo "gpt2.py não encontrado"; exit 1; }

# ====== EXECUTAR COM ACCELERATE ======
accelerate launch \
  --multi_gpu \
  --num_processes ${SLURM_NTASKS} \
  /home/vberta/projects/def-aloise/vberta/Paper3/gpt2.py
