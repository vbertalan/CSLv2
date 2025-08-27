#!/bin/bash
#SBATCH --job-name=qwen3-train
#SBATCH --nodes=1
#SBATCH --gres=gpu:4              # 4 GPUs no mesmo nó
#SBATCH --ntasks-per-node=4       # 1 processo por GPU
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=vbertalan@gmail.com
#SBATCH --mail-type=ALL

echo "=== Iniciando treino Qwen3 em $(hostname) ==="
nvidia-smi || true

# ====== MÓDULOS ======
module load StdEnv/2023
module load scipy-stack arrow
# Se o PyTorch já vem com CUDA runtime, não carregar 'cuda'

# ====== AMBIENTE VENV ======
source /home/vberta/projects/def-aloise/vberta/vbertapy/bin/activate

# ====== PATHS ======
PROJ_DIR=/home/vberta/projects/def-aloise/vberta/Paper3
SCRIPT=$PROJ_DIR/qwen.py
DATA=$PROJ_DIR/logs/part_3.log
OUT=$PROJ_DIR/qwen_part_3

# ====== CACHES ======
export HF_HOME="$SLURM_TMPDIR/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
mkdir -p "$HF_DATASETS_CACHE"

# ====== THREADS ======
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
ulimit -n 4096

# ====== NCCL ======
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29400
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# ====== CHECAGENS ======
cd "$PROJ_DIR" || { echo "Erro: diretório $PROJ_DIR não encontrado"; exit 1; }
ls -l "$SCRIPT" || { echo "Erro: script $SCRIPT não encontrado"; exit 1; }
ls -l "$DATA"   || { echo "Erro: dataset $DATA não encontrado"; exit 1; }

# ====== CONFIG DO ACCELERATE ======
ACCEL_YAML=$SLURM_TMPDIR/accelerate.yaml
cat > "$ACCEL_YAML" <<'YAML'
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
mixed_precision: bf16
num_processes: 4
num_machines: 1
machine_rank: 0
gpu_ids: all
YAML

# ====== HÍPERPARÂMETROS ======
VOCAB_SIZE=11000
BLOCK_SIZE=768
EPOCHS=20
BATCH_PER_GPU=20
GRAD_ACCUM=2
LR=1e-4

# ====== TREINO ======
accelerate launch --config_file "$ACCEL_YAML" "$SCRIPT" \
  --input_file "$DATA" \
  --out_dir "$OUT" \
  --vocab_size $VOCAB_SIZE \
  --block_size $BLOCK_SIZE \
  --epochs $EPOCHS \
  --batch_size $BATCH_PER_GPU \
  --grad_accum $GRAD_ACCUM \
  --lr $LR