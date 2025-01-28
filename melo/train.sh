CONFIG=$1
GPUS=$2
MODEL_NAME=$(basename "$(dirname $CONFIG)")

PORT=10902

torchrun --nproc_per_node=$GPUS \
        --master_port=$PORT \
    train.py --c $CONFIG --model $MODEL_NAME 