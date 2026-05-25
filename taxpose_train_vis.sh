# export WANDB_DISABLED=true
CONFIG=$1
CKPT_PATH=$2
## use ./launch.sh local 0 $command to run on local machine
ROOT_DIR=/home/yan/pose_estimation/taxpose/
cd $ROOT_DIR

# export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:512
export PYTEST_CURRENT_TEST=True

# train taxpose network
python "./vis_taxpose.py" \
    --config-name $CONFIG \
    job_type="train_taxpose" \
    model.freeze_embnn=True \
    model.encoder.name=dgcnn \
    model.norm=LN \
    wandb.name='vis_taxpose' \
    'model.pretraining.action.ckpt_path="logs/pretrain_embedding/2026-04-09/18-15-28/checkpoints/epoch=369-step=23310-train_loss=0.54-weights-only.ckpt"' \
    'model.pretraining.anchor.ckpt_path="logs/pretrain_embedding/2026-04-09/18-15-28/checkpoints/epoch=369-step=23310-train_loss=0.54-weights-only.ckpt"' \
    wandb.offline=True \
    resume_ckpt=$CKPT_PATH \
