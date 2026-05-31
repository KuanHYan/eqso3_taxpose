# export WANDB_DISABLED=true
GPU_INDEX=$1
SCHED=$2
WANDB_NAME=$3
CONFIG=$4
RESUME_CKPT=$5
## use ./launch.sh local 0 $command to run on local machine
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR="${SCRIPT_DIR}/"
cd "$ROOT_DIR" || exit 1

export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:512
export PYTEST_CURRENT_TEST="$TEST_MODE"

# train taxpose network
bash "./launch.sh" local $GPU_INDEX \
    python "./scripts/train_residual_flow_debug.py" \
    --config-name $CONFIG \
    job_type="train_taxpose" \
    data_root="${ROOT_DIR}data/pair_models" \
    training.max_epochs=135 \
    training.batch_size=32 \
    training.lr=0.0002 \
    training.min_lr=0.00002 \
    training.warmup_ratio=0.01 \
    training.precision=32 \
    training.scheduler=$SCHED \
    dataset@dm=tax_pose \
    dm.train_dset.demo_dset.num_demo=1024 \
    dm.train_dset.dataset_size=32000\
    model.freeze_embnn=True \
    model.encoder.name=raw_dgcnn \
    model.encoder.norm=BN \
    model.norm=LN \
    model.head_bias=False \
    model.dropout=0.3 \
    wandb.name=$WANDB_NAME \
    'model.pretraining.action.ckpt_path="${ROOT_DIR}logs/pretrain_embedding/best_cpkg/new_dgcnn_BN_509.ckpt"' \
    'model.pretraining.anchor.ckpt_path="${ROOT_DIR}logs/pretrain_embedding/best_cpkg/new_dgcnn_BN_509.ckpt"' \
    wandb.offline=True \
    training.displace_loss_weight=1 \
    training.direct_correspondence_loss_weight=1 \
    training.consistency_loss_weight=0.1 \
    'resume_ckpt="logs/train_taxpose/2026-04-30/17-04-43/checkpoints/last.ckpt"' \
    # wandb.project="tax-pose" \
    # wandb.save_dir="${ROOT_DIR}pretrain_embedding" \