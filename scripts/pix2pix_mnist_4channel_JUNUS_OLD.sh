set -ex
--dataroot datasets/mnist_4channel/AB --name mnist_4_channel_bs64_200dec --batch_size 64 --n_epochs 100 --n_epochs_decay 200 --model pix2pix --direction AtoB --input_nc 4 --output_nc 4 --no_flip

#--preprocess none