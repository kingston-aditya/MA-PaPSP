#!/bin/bash

# custom config
DATA="/data/datasets"
TRAINER=ZeroshotCLIP
DATASET=$1
CFG=$2  # rn50, rn101, vit_b32 or vit_b16

python /data/aditya/Dassl.pytorch/gen_main.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file /data/aditya/Dassl.pytorch/dassl/confil/datasets/${DATASET}.yaml \
--config-file /data/aditya/Dassl.pytorch/dassl/confil/trainers/${CFG}.yaml \
--eval-only