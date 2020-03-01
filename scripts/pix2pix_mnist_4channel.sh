set -ex
python train.py --dataroot ./datasets/mnist_4channel/AB --name mnist_4channel_unet128_200dec --model pix2pix --netG unet_128 --direction AtoB --input_nc 4 --output_nc 4 --batch_size 64 --n_epochs 100 --n_epochs_decay 200

#--preprocess none
