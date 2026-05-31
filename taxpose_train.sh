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
    python "./scripts/train_residual_flow.py" \
    --config-name $CONFIG \
    job_type="train_taxpose" \
    data_root="${ROOT_DIR}data/pair_models" \
    training.image_logging_period=1000 \
    training.log_every_n_steps=100 \
    training.check_val_every_n_epoch=1 \
    training.max_epochs=502 \
    training.end_lr_ratio=1.0 \
    training.batch_size=16 \
    training.lr=0.0001 \
    training.min_lr=0.00001 \
    training.warmup_ratio=0.05 \
    training.precision='32' \
    training.scheduler=$SCHED \
    training.consistency_loss_weight=0.1 \
    training.res_smooth_loss_weight=0.0 \
    dataset@dm=tax_pose \
    dm.train_dset.demo_dset.num_demo=1024 \
    dm.train_dset.dataset_size=3200 \
    model.freeze_embnn=True \
    model.dropout=0.1 \
    model.pos_encoding=False \
    model.encoder.name=raw_dgcnn \
    model.encoder.norm=BN \
    model.head.norm=LN \
    model.head.head_bias=False \
    model.head.head_type=transformer \
    model.head.residual_on=True \
    wandb.name=$WANDB_NAME \
    wandb.entity=yankunh27-zhejiang-university \
    model.pretraining.action.ckpt_path="${ROOT_DIR}logs/pretrain_embedding/best_cpkg/new_dgcnn_BN_509.ckpt" \
    model.pretraining.anchor.ckpt_path="${ROOT_DIR}logs/pretrain_embedding/best_cpkg/new_dgcnn_BN_509.ckpt" \
    wandb.offline=False \
    debug=False \
    resume_ckpt=$RESUME_CKPT \
    # wandb.project="tax-pose" \
    # wandb.save_dir="${ROOT_DIR}pretrain_embedding" \
