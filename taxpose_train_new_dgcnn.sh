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
s
ENCODEING=False

# train taxpose network
bash "./launch.sh" local $GPU_INDEX \
    python "./scripts/train_residual_flow.py" \
    --config-name $CONFIG \
    job_type="train_taxpose" \
    data_root="${ROOT_DIR}data/pair_models" \
    training.max_epochs=500 \
    training.check_val_every_n_epoch=1 \
    training.batch_size=44 \
    training.lr=0.00013 \
    training.min_lr=0.000013 \
    training.warmup_ratio=0.05 \
    training.precision='32' \
    training.scheduler=$SCHED \
    dataset@dm=tax_pose \
    dm.train_dset.demo_dset.num_demo=1024 \
    dm.train_dset.dataset_size=6400 \
    model.freeze_embnn=True \
    model.dropout=0.1 \
    model.pos_encoding=$ENCODEING \
    model.encoder.name=vae_dgcnn \
    model.encoder.norm=BN \
    model.encoder.emb_dims=256 \
    model.encoder.pos_encoding=$ENCODEING \
    model.encoder.output_num=1024 \
    model.head.head_type=residual \
    model.head.project_corrs=True \
    model.head.norm=LN \
    model.head.head_bias=False \
    model.head.residual_on=True \
    model.head.pred_weight=True \
    wandb.name=$WANDB_NAME \
    model.pretraining.action.ckpt_path="${ROOT_DIR}logs/pretrain_embedding/best_cpkg/VAE_dgcnn.ckpt" \
    model.pretraining.anchor.ckpt_path="${ROOT_DIR}logs/pretrain_embedding/best_cpkg/VAE_dgcnn.ckpt" \
    wandb.offline=True \
    debug=False \
    resume_ckpt=$RESUME_CKPT
