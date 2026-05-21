# export WANDB_DISABLED=true
GPU_INDEX=$1
TEST_MODE=$2
SCHED=$3
WANDB_NAME=$4
CONFIG=$5
EVAL=$6
RESUME_CKPT=$7
## use ./launch.sh local 0 $command to run on local machine
ROOT_DIR=/home/yan/pose_estimation/taxpose/
cd $ROOT_DIR

export WANDB_SSL_VERIFY=0
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:512
export PYTEST_CURRENT_TEST=$TEST_MODE

ENCODEING=False


bash "./launch.sh" local $GPU_INDEX \
    python "./scripts/train_residual_flow.py" \
    --config-name train_ndf \
    job_type="train_taxpose" \
    data_root="/home/yan/EmbodiedAgent/generate_data/pair_models/point_cloud" \
    training.max_epochs=500 \
    training.check_val_every_n_epoch=1 \
    training.batch_size=16 \
    training.lr=0.00015 \
    training.min_lr=0.000015 \
    training.warmup_ratio=0.05 \
    training.precision='32' \
    training.scheduler=linear \
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
    model.head.head_norm=LN \
    model.head.head_bias=False \
    model.head.residual_on=True \
    model.head.pred_weight=True \
    wandb.name=256_query_act_in_head \
    'model.pretraining.action.ckpt_path="logs/pretrain_embedding/best_cpkg/VAE_dgcnn.ckpt"' \
    'model.pretraining.anchor.ckpt_path="logs/pretrain_embedding/best_cpkg/VAE_dgcnn.ckpt"' \
    wandb.offline=True \
    debug=False \
    eval=$EVAL \
    resume_ckpt=$RESUME_CKPT
