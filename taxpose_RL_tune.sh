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

# export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:512
export PYTEST_CURRENT_TEST=$TEST_MODE

# train taxpose network
bash "./launch.sh" local $GPU_INDEX \
    python "./scripts/train_rl_tune.py" \
    --config-name $CONFIG \
    job_type="rl_tune" \
    data_root="${ROOT_DIR}data/pair_models" \
    training.max_epochs=500 \
    training.check_val_every_n_epoch=1 \
    training.batch_size=4 \
    training.lr=0.000025 \
    training.min_lr=0.0000025 \
    training.warmup_ratio=0.01 \
    training.precision='32' \
    training.scheduler=$SCHED \
    dataset@dm=tax_pose \
    dm.train_dset.demo_dset.num_demo=1024 \
    dm.train_dset.dataset_size=6400 \
    dm.train_dset.anchor_rot_sample_method=axis_angle \
    dm.train_dset.anchor_rotation_variance=3.141592653589793 \
    model.freeze_embnn=True \
    model.dropout=0.1 \
    model.encoder.name=raw_dgcnn \
    model.encoder.emb_dims=512 \
    model.encoder.norm=BN \
    model.encoder.output_num=1024 \
    model.encoder.dropout=0.3 \
    model.head.head_type=rl_residual \
    model.head.project_corrs=True \
    model.head.project_corrs_mode="vn" \
    model.head.norm=LN \
    model.head.head_bias=False \
    model.head.residual_on=True \
    model.head.pred_weight=True \
    rl.reward_model_path="${ROOT_DIR}logs/rl_reward/2026-05-24/10-51-49/checkpoints/last.ckpt" \
    rl.base_model_path="${ROOT_DIR}logs/train_taxpose/best_ckpt/vn_Wab_wo_TFhead_6400dz.ckpt" \
    rl.group=32 \
    rl.update_base_every=5 \
    rl.kl_coef=0.5 \
    rl.clip_eps=0.2 \
    wandb.name=$WANDB_NAME \
    wandb.entity=yankunh27-zhejiang-university \
    wandb.offline=False \
    debug=False \
    resume_ckpt=$RESUME_CKPT \
    # wandb.project="tax-pose" \
    # wandb.save_dir="${ROOT_DIR}pretrain_embedding" \
