# export WANDB_DISABLED=true
GPU_INDEX=$1
TEST_MODE=$2
CKPT_PATH=$3
## use ./launch.sh local 0 $command to run on local machine
ROOT_DIR=/home/yan/pose_estimation/taxpose/
cd $ROOT_DIR

export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:512
export PYTEST_CURRENT_TEST=$TEST_MODE

# Pretrain embeddings network
bash "./launch.sh" local $GPU_INDEX \
    python "./scripts/pretrain_embedding.py" \
    --config-name "pretraining" \
    job_type="pretrain_embedding" \
    data_root="/home/yan/EmbodiedAgent/generate_data/pair_models/point_cloud" \
    training.batch_size=16 \
    training.lr=1e-3 \
    training.precision=32 \
    training.scheduler='milestone' \
    training.epochs=600 \
    dataset=custom_dataset \
    wandb.offline=False \
    # resume_ckpt=$CKPT_PATH \
    # wandb.project="tax-pose" \
    # wandb.save_dir="${ROOT_DIR}pretrain_embedding" \