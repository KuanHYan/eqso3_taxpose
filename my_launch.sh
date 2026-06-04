# Get the first argument:
PLATFORM=$1

# Get the second argument:
GPU_INDEX=$2
shift
shift

# Get the third argument:
COMMAND=$@


echo Platform: $PLATFORM
echo GPU Index: $GPU_INDEX
echo Command: $COMMAND

# If the platform is "local-docker", then we need to use docker to run the command.
if [ $PLATFORM == "local-docker" ]; then
    echo "Running locally with docker"
    docker run -it \
        --network host \
        --gpus $GPU_INDEX \
        -e WANDB_BASE_URL=http://localhost:8080 \
        -e WANDB_API_KEY=local-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 \
        -v /data/yan/pose_dataset/pair_models:/root/pairpose/data \
        -v /data/yan/pose_ck_dir/taxpose/logs:/root/pairpose/logs \
        -v /home/yan/pose_estimation/taxpose:/root/pairpose \
        pair-pose:v0.1 \
        $COMMAND \
        log_dir=/root/pairpose/logs \
        data_root=/root/pairpose/data \

elif [ $PLATFORM == "cloud-featurize" ]; then
    echo "Running with cloud-featurize docker"
    docker run -it \
        --network host \
        --gpus all \
        --shm-size=48g \
        -e WANDB_BASE_URL=http://localhost:8080 \
        -e WANDB_API_KEY=local-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 \
        -v /home/featurize/data/ideal_pair_models:/root/pairpose/data \
        -v /home/featurize/work/taxpose/logs:/root/pairpose/logs \
        -v /home/featurize/work/eqso3_taxpose:/root/pairpose \
        crpi-6he4t9ttrgr1us6h.cn-hangzhou.personal.cr.aliyuncs.com/cross-pose/docker4train:v0.2 \
        $COMMAND \
        log_dir=/root/pairpose/logs \
        data_root=/root/pairpose/data \

elif [ $PLATFORM == "cloud-gongji" ]; then
    echo "Running with cloud-gongji docker"
    docker run -it \
        --network host \
        --gpus all \
        --shm-size=48g \
        -e WANDB_BASE_URL=http://localhost:8080 \
        -e WANDB_API_KEY=local-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 \
        crpi-6he4t9ttrgr1us6h.cn-hangzhou.personal.cr.aliyuncs.com/cross-pose/docker4train:v0.2 \
        $COMMAND \
        log_dir=/root/pairpose/logs \
        data_root=/root/pairpose/data \

elif [ $PLATFORM == "local" ]; then
    echo "Running locally"

    CUDA_VISIBLE_DEVICES=$GPU_INDEX \
    $COMMAND

else
    echo "Platform not recognized"
fi
