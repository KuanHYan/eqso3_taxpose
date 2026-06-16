GPU_INDEX=$1
NUM_GPUS=$2
GRAD_ACC=$3
TEST_MODE=$4
WANDB_NAME=$5
RESUME_CKPT=$6
## use ./launch.sh local 0 $command to run on local machine
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR="${SCRIPT_DIR}/"
cd "$ROOT_DIR" || exit 1

# export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:512
export PYTEST_CURRENT_TEST=$TEST_MODE

# train taxpose network
bash ./my_launch.sh local $GPU_INDEX \
    python "./scripts/train_rl_tune.py" \
    --config-name train_ndf \
    job_type="rl_tune" \
    data_root="${ROOT_DIR}data/ideal_pair_models" \
    training.max_epochs=500 \
    training.check_val_every_n_epoch=1 \
    training.batch_size=16 \
    training.lr=1e-6 \
    training.min_lr=1e-7 \
    training.warmup_ratio=0.02 \
    training.weight_decay=0.0 \
    training.precision='32' \
    training.num_gpus=$NUM_GPUS \
    training.accumulate_grad_batches=$GRAD_ACC \
    training.scheduler=constant \
    dataset@dm=tax_pose \
    dm.train_dset.demo_dset.num_demo=6000 \
    dm.train_dset.dataset_size=6000 \
    dm.train_dset.anchor_rot_sample_method=axis_angle \
    dm.train_dset.anchor_rotation_variance=3.141592653589793 \
    model.freeze_embnn=True \
    model.dropout=0.1 \
    model.n_blocks=1 \
    model.cycle=False \
    model.model_type=rl_flow \
    model.attn_mode="torch_attn" \
    model.encoder.name=raw_dgcnn \
    model.encoder.emb_dims=512 \
    model.encoder.norm=BN \
    model.encoder.output_num=1024 \
    model.encoder.dropout=0.1 \
    model.head.head_type=rl_residual \
    model.head.project_corrs=True \
    model.head.project_corrs_mode="moe" \
    model.head.norm=LN \
    model.head.head_bias=False \
    model.head.residual_on=True \
    model.head.pred_weight=True \
    model.head.reparam=True \
    rl.reward_model_path="/home/yan/pose_estimation/taxpose/trained_models/reward_w.ckpt" \
    rl.base_model_path="/home/yan/pose_estimation/taxpose/logs/train_taxpose/2026-06-13/01-05-28/checkpoints/last.ckpt" \
    rl.group=64 \
    rl.update_base_every=100 \
    rl.kl_coef=0.05 \
    rl.clip_eps=0.2 \
    rl.grpo_iter=1 \
    wandb.name=$WANDB_NAME \
    wandb.offline=False \
    debug=False \
    eval=False
