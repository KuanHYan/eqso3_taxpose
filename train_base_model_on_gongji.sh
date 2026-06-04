# export WANDB_DISABLED=true
RESUME_CKPT=$0
## use ./launch.sh local 0 $command to run on local machine
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR="${SCRIPT_DIR}/"
cd "$ROOT_DIR" || exit 1

# export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:512
ENCODEING=False

CUDA_VISIBLE_DEVICES="0"

python "./scripts/train_residual_flow.py" \
    --config-name train_ndf \
    job_type="train_taxpose" \
    data_root="${ROOT_DIR}data_default/ideal_pair_models" \
    training.max_epochs=500 \
    training.check_val_every_n_epoch=1 \
    training.batch_size=8 \
    training.lr=0.000025 \
    training.min_lr=0.0000025 \
    training.warmup_ratio=0.05 \
    training.precision='32' \
    training.scheduler=linear \
    training.num_gpus=4 \
    dataset@dm=tax_pose \
    dm.train_dset.demo_dset.num_demo=6000 \
    dm.train_dset.dataset_size=32000 \
    dm.train_dset.anchor_rot_sample_method=axis_angle \
    dm.train_dset.anchor_rotation_variance=3.141592653 \
    model.freeze_embnn=True \
    model.dropout=0.1 \
    model.n_blocks=2 \
    model.pos_encoding=$ENCODEING \
    model.encoder.name=raw_dgcnn \
    model.encoder.norm=BN \
    model.encoder.emb_dims=512 \
    model.encoder.pos_encoding=$ENCODEING \
    model.encoder.output_num=1024 \
    model.head.head_type=transformer \
    model.head.project_corrs=True \
    model.head.project_corrs_mode="moe" \
    model.head.norm=LN \
    model.head.head_bias=False \
    model.head.residual_on=True \
    model.head.pred_weight=True \
    wandb.name="gongji_trainning" \
    model.pretraining.action.ckpt_path="trained_models/5000pts_dgcnn.ckpt" \
    model.pretraining.anchor.ckpt_path="trained_models/5000pts_dgcnn.ckpt" \
    wandb.offline=True \
    debug=False \
