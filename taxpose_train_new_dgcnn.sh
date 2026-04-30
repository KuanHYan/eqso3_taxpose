# export WANDB_DISABLED=true
GPU_INDEX=$1
TEST_MODE=$2
SCHED=$3
WANDB_NAME=$4
CONFIG=$5
RESUME_CKPT=$6
## use ./launch.sh local 0 $command to run on local machine
ROOT_DIR=/home/yan/pose_estimation/taxpose/
cd $ROOT_DIR

export WANDB_SSL_VERIFY=0
# export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:512
export PYTEST_CURRENT_TEST=$TEST_MODE

# train taxpose network
bash "./launch.sh" local $GPU_INDEX \
    python "./scripts/train_residual_flow.py" \
    --config-name $CONFIG \
    job_type="train_taxpose" \
    data_root="/home/yan/EmbodiedAgent/generate_data/pair_models/point_cloud" \
    training.max_epochs=500 \
    training.check_val_every_n_epoch=1 \
    training.batch_size=32 \
    training.lr=0.0002 \
    training.min_lr=0.000002 \
    training.warmup_ratio=0.01 \
    training.precision='32' \
    training.scheduler=$SCHED \
    dataset@dm=tax_pose \
    dm.train_dset.demo_dset.num_demo=1024 \
    dm.train_dset.dataset_size=64000 \
    model.freeze_embnn=True \
    model.encoder.name=raw_dgcnn \
    model.encoder.norm=BN \
    model.head_norm=LN \
    model.head_bias=False \
    model.dropout=0.1 \
    wandb.name=$WANDB_NAME \
    'model.pretraining.action.ckpt_path="logs/pretrain_embedding/best_cpkg/new_dgcnn_BN_509.ckpt"' \
    'model.pretraining.anchor.ckpt_path="logs/pretrain_embedding/best_cpkg/new_dgcnn_BN_509.ckpt"' \
    wandb.offline=False \
    resume_ckpt=$RESUME_CKPT \
    # wandb.project="tax-pose" \
    # wandb.save_dir="${ROOT_DIR}pretrain_embedding" \
