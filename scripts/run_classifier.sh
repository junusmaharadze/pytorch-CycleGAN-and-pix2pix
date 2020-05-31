#!/bin/bash

mode="$1"

# Check if any argument is passed
if [[ $# -eq 0 ]]
then
    echo "No arguments supplied. Pass train or test as an argument"
    exit
fi

echo "Mode=$mode"
# echo

MODEL=resnet18
DATA_DIR=./datasets/xBD_datasets/xBD_polygons_AB
LABELS_FILE=./datasets/xBD_datasets/xBD_polygons_AB_csv/satellite_AB_labels.txt
CHECKPOINT_NAME=resnet18_checkpoint
BATCH_SIZE=16

if [ "$mode" == "train" ]
then
    set -ex
    python classifiers/train.py \
    --model $MODEL \
    --data_dir $DATA_DIR \
    --labels_file $LABELS_FILE \
    --num_epochs 100 \
    --batch_size $BATCH_SIZE \
    --checkpoint_name $CHECKPOINT_NAME

elif [ "$mode" == "test" ]
then
    set -ex
    python classifiers/test.py \
    --model $MODEL \
    --data_dir $DATA_DIR \
    --labels_file $LABELS_FILE \
    --batch_size $BATCH_SIZE \
    --checkpoint_name $CHECKPOINT_NAME

else
    echo "Invalid argument. Use train or test"
fi
