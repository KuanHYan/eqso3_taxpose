# export WANDB_DISABLED=true
GPU_INDEX=$1
WANDB_NAME=$2
## use ./launch.sh local 0 $command to run on local machine
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR="${SCRIPT_DIR}/"
cd "$ROOT_DIR" || exit 1

export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:512
export PYTEST_CURRENT_TEST="$TEST_MODE"

# Pretrain embeddings network
bash "./launch.sh" local $GPU_INDEX \
    python "./scripts/pretrain_embedding_debug.py" \
    --config-name "pretraining" \
    job_type="pretrain_embedding" \
    data_root="${ROOT_DIR}data/pair_models" \
    training.batch_size=16 \
    training.lr=1e-3 \
    training.precision=32 \
    training.scheduler='constant' \
    training.epochs=5 \
    encoder.name=dgcnn \
    wandb.name=$WANDB_NAME \
    dataset=custom_dataset \
    wandb.offline=True \
    # resume_ckpt=/home/yan/pose_estimation/taxpose/logs/pretrain_embedding/2026-04-09/18-15-28/checkpoints/last.ckpt \
    # wandb.project="tax-pose" \
    # wandb.save_dir="${ROOT_DIR}pretrain_embedding" \