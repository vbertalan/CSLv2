#!/bin/bash
#SBATCH --job-name=gpt2
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4        # 1 processo por GPU
#SBATCH --gres=gpu:4               # 4 GPUs requisitadas
#SBATCH --cpus-per-task=6          # Ajuste conforme necessário
#SBATCH --time=24:00:00
#SBATCH --output=%j-gpt2.out
#SBATCH --mail-user=vbertalan@gmail.com
#SBATCH --mail-type=ALL

echo "Training GPT2..."
echo "Starting multi-GPU training job on $(hostname)"
echo "GPUs available on this node:"
nvidia-smi

# === ENV SETUP ===
module load cuda scipy-stack arrow
source /home/vberta/projects/def-aloise/vberta/vbertapy/bin/activate

export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# === ACESSAR DIRETÓRIO DO SCRIPT ===
cd /home/vberta/projects/def-aloise/vberta/Paper3/
echo "Accessing folder..."

# === EXECUTAR COM ACCELERATE ===
accelerate launch --multi_gpu gpt2.py
