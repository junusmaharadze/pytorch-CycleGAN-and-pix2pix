set -ex
python train.py \
--dataroot ./datasets/xBD_datasets/xBD_polygons_AB/AB/ \
--labels_file ./datasets/xBD_datasets/xBD_polygons_AB_csv/satellite_AB_labels.txt \
--name satellite_pix2pix_bs8 \
--model pix2pix \
--direction AtoB \
--input_nc 3 \
--output_nc 3 \
--batch_size 32 \
--n_epochs 10 \
--epoch_count 4 \
--n_epochs_decay 10 \
--model pix2pix \
--no_flip \
--netG unet_128 \
--load_size 142 \
--crop_size 128 \
--num_threads 1
