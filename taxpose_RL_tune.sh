# export WANDB_DISABLED=true
PLATFORM=$1
GPU_INDEX=$2
TEST_MODE=$3
SCHED=$4
WANDB_NAME=$5
RESUME_CKPT=$6
## use ./launch.sh local 0 $command to run on local machine
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR="${SCRIPT_DIR}/"
cd "$ROOT_DIR" || exit 1

# export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:512
export PYTEST_CURRENT_TEST=$TEST_MODE

# train taxpose network
bash ./my_launch.sh $PLATFORM $GPU_INDEX \
    python "./scripts/train_rl_tune.py" \
    --config-name train_ndf \
    job_type="rl_tune" \
    data_root="${ROOT_DIR}data/ideal_pair_models" \
    training.max_epochs=500 \
    training.check_val_every_n_epoch=1 \
    training.batch_size=6 \
    training.lr=0.00005 \
    training.min_lr=0.0000002 \
    training.warmup_ratio=0.1 \
    training.precision='32' \
    training.num_gpus=1 \
    training.scheduler=$SCHED \
    dataset@dm=tax_pose \
    dm.train_dset.demo_dset.num_demo=1024 \
    dm.train_dset.dataset_size=6400 \
    dm.train_dset.anchor_rot_sample_method=axis_angle \
    dm.train_dset.anchor_rotation_variance=3.141592653589793 \
    model.freeze_embnn=True \
    model.dropout=0.1 \
    model.n_blocks=1 \
    model.encoder.name=raw_dgcnn \
    model.encoder.emb_dims=512 \
    model.encoder.norm=BN \
    model.encoder.output_num=1024 \
    model.encoder.dropout=0.1 \
    model.head.head_type=rl_transformer \
    model.head.project_corrs=True \
    model.head.project_corrs_mode="moe" \
    model.head.norm=LN \
    model.head.head_bias=False \
    model.head.residual_on=True \
    model.head.pred_weight=True \
    rl.reward_model_path="logs/best_cpkg/reward_w.ckpt" \
    rl.base_model_path="logs/train_taxpose/2026-05-30/10-42-04/checkpoints/last.ckpt" \
    rl.group=8 \
    rl.update_base_every=5 \
    rl.kl_coef=0.02 \
    rl.clip_eps=0.2 \
    wandb.name=$WANDB_NAME \
    wandb.offline=True \
    debug=False \
    eval=False
