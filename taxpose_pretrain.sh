# export WANDB_DISABLED=true
GPU_INDEX=$1
TEST_MODE=$2
WANDB_NAME=$3
## use ./launch.sh local 0 $command to run on local machine
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR="${SCRIPT_DIR}/"
cd "$ROOT_DIR" || exit 1

export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:512
export PYTEST_CURRENT_TEST="$TEST_MODE"

# Pretrain embeddings network
bash "./launch.sh" local $GPU_INDEX \
    python "./scripts/pretrain_embedding.py" \
    --config-name "pretraining" \
    job_type="pretrain_embedding" \
    data_root="${ROOT_DIR}data/ideal_pair_models" \
    training.batch_size=25 \
    training.lr=1.0e-4 \
    training.precision=32 \
    training.scheduler='constant' \
    training.epochs=400 \
    encoder.name=dgcnn_group \
    encoder.emb_dims=512 \
    encoder.knn=20 \
    encoder.norm=BN \
    encoder.output_num=512 \
    wandb.name=$WANDB_NAME \
    dataset=custom_dataset \
    dataset.train_dset.data_size=16000 \
    wandb.offline=False \
    # resume_ckpt=$CKPT_PATH \
    # wandb.project="tax-pose" \
    # wandb.save_dir="${ROOT_DIR}pretrain_embedding" \