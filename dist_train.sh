#!/usr/bin/env bash

GPUS=$2
PORT=${PORT:-28500}
#PORT=${PORT:-12345}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/main_pedtr.py
