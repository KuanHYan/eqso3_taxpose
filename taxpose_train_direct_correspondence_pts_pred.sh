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
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:512
export PYTEST_CURRENT_TEST=$TEST_MODE

# train taxpose network
bash "./launch.sh" local $GPU_INDEX \
    python "./scripts/train_residual_flow.py" \
    --config-name $CONFIG \
    job_type="train_taxpose" \
    seed=0 \
    data_root="/data/yan/pose_dataset/pair_models" \
    training.max_epochs=500 \
    training.check_val_every_n_epoch=1 \
    training.batch_size=32 \
    training.lr=0.0001 \
    training.min_lr=0.00001 \
    training.warmup_ratio=0.05 \
    training.precision='32' \
    training.scheduler=$SCHED \
    dataset@dm=tax_pose \
    dm.train_dset.demo_dset.num_demo=1024 \
    dm.train_dset.anchor_rot_sample_method=axis_angle_uniform_z \
    dm.train_dset.anchor_rotation_variance=1e-5 \
    dm.train_dset.dataset_size=6400 \
    model.freeze_embnn=True \
    model.dropout=0.1 \
    model.encoder.name=raw_dgcnn \
    model.encoder.emb_dims=512 \
    model.encoder.norm=BN \
    model.encoder.output_num=1024 \
    model.head.head_type=residual \
    model.head.project_corrs=True \
    model.head.project_corrs_mode="vn" \
    model.head.norm=LN \
    model.head.head_bias=False \
    model.head.residual_on=True \
    model.head.pred_weight=True \
    wandb.name=$WANDB_NAME \
    'model.pretraining.action.ckpt_path="logs/pretrain_embedding/best_cpkg/new_dgcnn_BN_509.ckpt"' \
    'model.pretraining.anchor.ckpt_path="logs/pretrain_embedding/best_cpkg/new_dgcnn_BN_509.ckpt"' \
    wandb.offline=False \
    debug=False \
    eval=True \
    resume_ckpt=$RESUME_CKPT \
    # wandb.project="tax-pose" \
    # wandb.save_dir="${ROOT_DIR}pretrain_embedding" \
