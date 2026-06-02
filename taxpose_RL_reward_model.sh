# export WANDB_DISABLED=true
GPU_INDEX=$1
TEST_MODE=$2
SCHED=$3
WANDB_NAME=$4
CONFIG=$5
RESUME_CKPT=$6
## use ./launch.sh local 0 $command to run on local machine
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR="${SCRIPT_DIR}/"
cd "$ROOT_DIR" || exit 1

export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:512
export PYTEST_CURRENT_TEST="$TEST_MODE"

# train taxpose network
bash "./launch.sh" local $GPU_INDEX \
    python "./scripts/train_rl_reward.py" \
    --config-name $CONFIG \
    job_type="rl_reward" \
    data_root="${ROOT_DIR}data/pair_models" \
    training.image_logging_period=1000 \
    training.log_every_n_steps=100 \
    training.check_val_every_n_epoch=1 \
    training.max_epochs=300 \
    training.end_lr_ratio=1.0 \
    training.batch_size=18 \
    training.lr=0.0001 \
    training.min_lr=0.000001 \
    training.warmup_ratio=0.05 \
    training.precision='32' \
    training.scheduler=$SCHED \
    dataset@dm=tax_pose \
    dm.train_dset.demo_dset.num_demo=1024 \
    dm.train_dset.dataset_size=32000 \
    dm.train_dset.anchor_rot_sample_method=axis_angle \
    dm.train_dset.anchor_rotation_variance=3.141592653589793 \
    model.freeze_embnn=False \
    model.dropout=0.1 \
    model.pos_encoding=False \
    model.encoder.name=raw_dgcnn \
    model.encoder.norm=BN \
    wandb.name=$WANDB_NAME \
    wandb.entity=yankunh27-zhejiang-university \
    wandb.offline=True \
    debug=False \
    resume_ckpt=$RESUME_CKPT \
    # wandb.project="tax-pose" \
    # wandb.save_dir="${ROOT_DIR}pretrain_embedding" \
