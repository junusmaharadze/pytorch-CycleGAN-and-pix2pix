set -ex
python classifiers/train.py \
--model resnet18 \
--num_epochs 100 \
--data_dir ./datasets/xBD_datasets/xBD_polygons_AB \
--labels_file ./datasets/xBD_datasets/xBD_polygons_AB_csv/satellite_AB_labels.txt \
--checkpoint_name resnet18_checkpoint \
