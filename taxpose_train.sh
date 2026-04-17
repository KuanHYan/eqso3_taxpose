# export WANDB_DISABLED=true
GPU_INDEX=$1
TEST_MODE=$2
SCHED=$3
WANDB_NAME=$4
CONFIG=$5
## use ./launch.sh local 0 $command to run on local machine
ROOT_DIR=/home/yan/pose_estimation/taxpose/
cd $ROOT_DIR

# export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:512
export PYTEST_CURRENT_TEST=TEST_MODE

# train taxpose network
bash "./launch.sh" local $GPU_INDEX \
    python "./scripts/train_residual_flow.py" \
    --config-name $CONFIG \
    job_type="train_taxpose" \
    data_root="/home/yan/EmbodiedAgent/generate_data/pair_models/point_cloud" \
    training.max_epochs=1500 \
    training.batch_size=16 \
    training.lr=0.0002 \
    training.min_lr=0.00002 \
    training.warmup_ratio=0.001 \
    training.precision=32 \
    training.scheduler=$SCHED \
    dataset@dm=tax_pose \
    dm.train_dset.demo_dset.num_demo=1024 \
    dm.train_dset.dataset_size=65000 \
    model.freeze_embnn=True \
    wandb.name=$WANDB_NAME \
    'model.pretraining.action.ckpt_path="logs/pretrain_embedding/2026-04-09/18-15-28/checkpoints/epoch=369-step=23310-train_loss=0.54-weights-only.ckpt"' \
    'model.pretraining.anchor.ckpt_path="logs/pretrain_embedding/2026-04-09/18-15-28/checkpoints/epoch=369-step=23310-train_loss=0.54-weights-only.ckpt"' \
    wandb.offline=True \
    resume_ckpt='logs/train_taxpose/2026-04-15/17-16-50/checkpoints/last.ckpt' \
    # wandb.project="tax-pose" \
    # wandb.save_dir="${ROOT_DIR}pretrain_embedding" \
