docker run -it --rm \
    --network host \
    --gpus all \
    -e WANDB_BASE_URL=http://localhost:8080 \
    -e WANDB_API_KEY=local-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 \
    -v /data/yan/pose_dataset:/opt/pairpose/data \
    -v /data/yan/pose_ck_dir/taxpose/logs:/opt/pairpose/logs \
    pair-pose:v0.1