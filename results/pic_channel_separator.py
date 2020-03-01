
import cv2
from PIL import Image
import numpy as np
import os
import regex as re


def separate_rgb_layer(mypath):
    readpath = os.path.join(mypath, 'test_latest/images')
    print(readpath)
    savepath = os.path.join(mypath, 'separated')
    for f in os.listdir(readpath):
        img = cv2.imread(os.path.join(readpath, f), cv2.IMREAD_UNCHANGED)
        rgb = img[:,:,0:3]
        layer = img[:,:,3]
        print(np.mean(layer))
        rgb = Image.fromarray(rgb)
        layer = Image.fromarray(layer, 'L')
        rgb.save(os.path.join(savepath, re.sub('.png', 'rgb.png', f)))
        layer.save(os.path.join(savepath, re.sub('.png', 'layer.png', f)))


if __name__ == '__main__':
    path = './mnist_4_channel_bs64_200dec/'
    separate_rgb_layer(path)
