set -ex
python train.py \
--dataroot ./datasets/xBD_datasets/xBD_polygons_AB/AB \
--name DamageGAN_initial \
--model pix2pix \
--direction AtoB \
--input_nc 3 \
--output_nc 3 \
--batch_size 128 \
--n_epochs 100 \
--epoch_count 0 \
--n_epochs_decay 100 \
--model pix2pix \
--netG unet_128 \
--load_size 142 \
--crop_size 128 \
--num_threads 12 \
--labels_file ./datasets/xBD_datasets/xBD_polygons_AB_csv/satellite_AB_labels.txt \
--intermediate_results_dir ./interim_results/
