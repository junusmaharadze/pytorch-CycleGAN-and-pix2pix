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

if [ "$mode" == "train" ]
then
    set -ex
    python classifiers/train.py \
    --model resnet18 \
    --data_dir ./datasets/xBD_datasets/xBD_polygons_AB \
    --labels_file ./datasets/xBD_datasets/xBD_polygons_AB_csv/satellite_AB_labels.txt \
    --num_epochs 100 \
    --batch_size 16 \
    --checkpoint_name resnet18_checkpoint

elif [ "$mode" == "test" ]
then
    set -ex
    python classifiers/test.py \
    --model resnet18 \
    --data_dir ./datasets/xBD_datasets/xBD_polygons_AB \
    --labels_file ./datasets/xBD_datasets/xBD_polygons_AB_csv/satellite_AB_labels.txt \
    --batch_size 16 \
    --checkpoint_name resnet18_checkpoint

else
    echo "Invalid argument. Use train or test"
fi
