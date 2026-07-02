# export WANDB_DISABLED=true
GPU_INDEX=$1
NUM_GPUS=$2
GRAD_ACC=$3
TEST_MODE=$4
WANDB_NAME=$5
EVAL=$6
RESUME_CKPT=$7
## use ./launch.sh local 0 $command to run on local machine
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR="${SCRIPT_DIR}/"
cd "$ROOT_DIR" || exit 1

# export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:512
export PYTEST_CURRENT_TEST=$TEST_MODE
ENCODEING=False
POINTS=500

bash "./my_launch.sh" local $GPU_INDEX \
    python "./scripts/train_residual_flow.py" \
    --config-name train_ndf \
    job_type="train_taxpose" \
    data_root="${ROOT_DIR}data/ideal_pair_models" \
    training.max_epochs=500 \
    training.check_val_every_n_epoch=1 \
    training.batch_size=10 \
    training.lr=0.6e-4 \
    training.min_lr=0.6e-5 \
    training.warmup_ratio=0.05 \
    training.weight_decay=1e-4 \
    training.precision='32' \
    training.scheduler=linear \
    training.num_gpus=$NUM_GPUS \
    training.accumulate_grad_batches=$GRAD_ACC \
    training.point_cloud_loss="MSE_sum" \
    training.displace_loss_weight=1.0 \
    training.direct_correspondence_loss_weight=1.0 \
    training.consistency_loss_weight=0.1 \
    dataset@dm=tax_pose \
    dm.test_folder=val_data \
    dm.train_dset.dataset_size=6400 \
    dm.train_dset.demo_dset.num_demo=6000 \
    dm.train_dset.demo_dset.min_num_points=1000 \
    dm.train_dset.demo_dset.occlusion_cfg.random_dropout=False \
    dm.train_dset.demo_dset.occlusion_cfg.anisotropic_scaling=False \
    dm.train_dset.demo_dset.occlusion_cfg.gaussian_noise=False \
    dm.train_dset.demo_dset.occlusion_cfg.occlusion_class=0 \
    dm.train_dset.anchor_rot_sample_method=axis_angle \
    dm.train_dset.anchor_rotation_variance=3.141592653 \
    model.model_type=two_stage_flow_transformer \
    model.freeze_embnn=False \
    model.dropout=0.1 \
    model.n_blocks=1 \
    model.attn_mode="linear" \
    model.pos_encoding=$ENCODEING \
    model.fine_tune=False \
    model.num_refine_steps=2 \
    model.refine_hidden_dim=128 \
    model.encoder.name=vn_dgcnn \
    model.encoder.norm=BN \
    model.encoder.emb_dims=384 \
    model.encoder.pos_encoding=False \
    model.encoder.output_num=$POINTS \
    model.encoder.down_ratio=2 \
    model.encoder.knn=20 \
    model.head.head_type=residual \
    model.head.project_corrs=True \
    model.head.project_corrs_mode="mlp" \
    model.head.norm=LN \
    model.head.head_bias=False \
    model.head.residual_on=True \
    model.head.pred_weight=True \
    model.head.reparam=False \
    wandb.name=$WANDB_NAME \
    wandb.offline=$EVAL \
    debug=True \
    eval=$EVAL \
    resume_ckpt=$RESUME_CKPT
exit 1
