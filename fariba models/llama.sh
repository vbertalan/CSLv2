#!/bin/bash
#SBATCH --job-name=tinyllama
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4       # ✅ 4 GPUs H100 completas
#SBATCH --ntasks-per-node=4          # 1 processo por GPU
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=vbertalan@gmail.com
#SBATCH --mail-type=ALL

echo "=== TinyLlama Fine-tuning Job ==="
echo "Job started on $(hostname)"
echo "Date: $(date)"
echo "==============================="

# ====== MÓDULOS ======
module load StdEnv/2023
module load python/3.10
module load cuda/12.1
module load arrow
module load scipy-stack

# ====== AMBIENTE VIRTUAL ======
source /home/vberta/projects/def-aloise/vberta/vbertapy/bin/activate

# ====== VARIÁVEIS ======
PROJ_DIR=/home/vberta/projects/def-aloise/vberta/Paper3/fariba
SCRIPT=${PROJ_DIR}/train_llama.py
export HF_HOME=${PROJ_DIR}/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export TORCH_HOME=${HF_HOME}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ====== NCCL (MULTI-GPU ESTABILIDADE) ======
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_IB_DISABLE=1

# ====== DEBUG DE GPUS ======
echo "GPUs disponíveis:"
nvidia-smi || true
echo "==============================="

# ====== EXECUÇÃO ======
cd $PROJ_DIR

echo "🚀 Iniciando treinamento TinyLlama..."
torchrun --nproc_per_node=4 $SCRIPT

echo "✅ Treinamento finalizado em $(date)"
