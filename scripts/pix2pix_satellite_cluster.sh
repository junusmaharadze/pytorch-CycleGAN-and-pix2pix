set -ex
input=$1
echo $input
# python train.py --dataroot ./datasets/mnist0123/AB --name mnist0123_pix2pix_bs8 --model pix2pix --direction AtoB --input_nc 3 --output_nc 3 --batch_size 8 --n_epochs 100 --n_epochs_decay 100 --model pix2pix --no_flip --load_size 84 --crop_size 64
python train.py --dataroot /disk/scratch/s1885912/datasets/datasets/satellite_AB/AB --name satellite_pix2pix_bs128 --model pix2pix --direction AtoB --input_nc 3 --output_nc 3 --batch_size 128 --n_epochs 100 --n_epochs_decay 100 --model pix2pix --no_flip --netG unet_128 --load_size 142 --crop_size 128 --display_id 0 --epoch_count 5 #--gpu_ids 0,1,2,3,4,5,6,7 --display_id 0
