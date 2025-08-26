#!/bin/bash
#SBATCH --job-name=neox-sent
#SBATCH --nodes=1
#SBATCH --gres=gpu:4                 # 4 GPUs no mesmo nó
#SBATCH --ntasks-per-node=4          # 1 processo por GPU
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=vbertalan@gmail.com
#SBATCH --mail-type=ALL

echo "Starting NeoX sentence-level LM training on $(hostname)"
echo "GPUs on this node:"
nvidia-smi || true

# ====== MÓDULOS ======
module load StdEnv/2023
module load scipy-stack arrow
# Se o seu PyTorch pip já inclui CUDA runtime, não carregue 'cuda' aqui.

# ====== AMBIENTE VENV ======
source /home/vberta/projects/def-aloise/vberta/vbertapy/bin/activate

# ====== PATHS ======
PROJ_DIR=/home/vberta/projects/def-aloise/vberta/Paper3
SCRIPT=${PROJ_DIR}/neox.py            # novo script com GPU/AMP/DDP
DATA=${PROJ_DIR}/logs/part_3.log                   # 1 frase por linha
OUT=${PROJ_DIR}/neox_sent                          # saída do modelo/tokenizer

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

# ====== NCCL/DDP – single-node ======
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29400
export ACCELERATE_USE_NCCL=1
export PYTORCH_DISTRIBUTED_BACKEND=nccl

# Configs NCCL estáveis em single-node
export NCCL_IB_DISABLE=1            # desliga IB em single-node
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1
unset NCCL_ASYNC_ERROR_HANDLING #  RETIRAR
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_TIMEOUT=7200

# Evitar fragmentação do CUDA allocator
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

echo "PWD: $(pwd)"
cd "$PROJ_DIR" || { echo "Diretório $PROJ_DIR não encontrado"; exit 1; }
ls -l "$SCRIPT" || { echo "Script não encontrado: $SCRIPT"; exit 1; }
ls -l "$DATA"   || { echo "Arquivo de dados não encontrado: $DATA"; exit 1; }

# ====== CONFIG DO ACCELERATE (evita avisos de defaults) ======
ACCEL_YAML=$SLURM_TMPDIR/accelerate_slurm_single.yaml
cat > "$ACCEL_YAML" <<'YAML'
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
mixed_precision: bf16
fsdp_config: {}
deepspeed_config: {}
num_processes: 4
num_machines: 1
machine_rank: 0
gpu_ids: all
downcast_bf16: 'no'
YAML
echo "Accelerate config em: $ACCEL_YAML"
echo "SLURM_NTASKS=${SLURM_NTASKS:-unset}"

# ====== HÍPERPARÂMETROS ======
VOCAB_SIZE=11000
BLOCK_SIZE=768      # ou 1024 se quiser cobrir p95
EPOCHS=3
BATCH_PER_GPU=20
GRAD_ACCUM=2
LR=5e-4

# ====== TREINO COM ACCELERATE ======
# Opção 1: chamada direta (funciona bem em single-node)
# Rode o treino
accelerate launch \
  --num_processes 3 \
  neox.py \
  --input_file logs/part_3.log \
  --out_dir part3_teste \
  --vocab_size 11000 \
  --block_size 768 \
  --epochs 3 \
  --batch_size 20 \
  --amp auto \
  --dataloader_workers 2