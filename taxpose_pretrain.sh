# export WANDB_DISABLED=true
GPU_INDEX=$1
TEST_MODE=$2
WANDB_NAME=$3
CKPT_PATH=$4
## use ./launch.sh local 0 $command to run on local machine
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR="${SCRIPT_DIR}/"
cd "$ROOT_DIR" || exit 1

export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:512
export PYTEST_CURRENT_TEST="$TEST_MODE"
ENCODEING=True
POINTS=1024
EMB_DIM=512
EVAl=False
# Pretrain embeddings network
bash "./my_launch.sh" local $GPU_INDEX \
    python "./scripts/pretrain_embedding.py" \
    --config-name "pretraining" \
    job_type="pretrain_embedding" \
    data_root="${ROOT_DIR}data/ideal_pair_models" \
    training.batch_size=16 \
    training.lr=1e-4 \
    training.precision=32 \
    training.scheduler='constant' \
    training.epochs=400 \
    encoder.name=dgcnn_group \
    encoder.down_layers=[1] \
    encoder.emb_dims=$EMB_DIM \
    encoder.knn=20 \
    encoder.norm=BN \
    encoder.output_num=$POINTS \
    encoder.pos_encoding=$ENCODEING \
    encoder.dropout=0.1 \
    wandb.name=$WANDB_NAME \
    dataset=custom_dataset \
    dataset.train_dset.data_size=18000 \
    dataset.train_dset.num_points=1024 \
    wandb.offline=$EVAl \
    eval=$EVAl \
    resume_ckpt=$CKPT_PATH \
    # wandb.project="tax-pose" \
    # wandb.save_dir="${ROOT_DIR}pretrain_embedding" \

## v1: 512 embedding 1024 points 0.101
## v2: 384 embedding 1024 points 0.090
## v3: 512 embedding 896 points  0.075
## v4: 512 embedding 768 points
## v5: 512 embedding 640 points
## v6: 384 embedding 896 points
## v7: 384 embedding 768 points
## v8: 384 embedding 640 points