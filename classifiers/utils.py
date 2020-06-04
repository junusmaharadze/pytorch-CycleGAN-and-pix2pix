import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from torchvision import models


def parse_arguments():
    parser = argparse.ArgumentParser('Train classifier')
    parser.add_argument('--model', dest='model', type=str, default='resnet18',
                        help='One of the models: resnet18|vgg4|densenet')
    parser.add_argument('--data_dir', dest='data_dir', type=str,
                        help='Path to the dataset containing the polygons and the labels textfile, eg: ./datasets/xBD_datasets/xBD_polygons_AB')
    parser.add_argument('--labels_file', dest='labels_file', type=str,
                        help='Path to the labels txt, eg: ./datasets/xBD_datasets/xBD_polygons_AB_csv/satellite_AB_labels.txt')
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16)
    parser.add_argument('--checkpoint_name', dest='checkpoint_name', type=str, default='resnet18_checkpoint',
                        help='The name of the best model\'s checkpoint to be used for inference')
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=0)
    args = parser.parse_args()

    for arg in vars(args):
        print('[%s] =' % arg, getattr(args, arg))

    return args


def initialize_model_and_params(arg_model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if arg_model == 'resnet18':
        model = models.resnet18(pretrained=False, progress=True)
    # Add more models here

    print('Device:', device)
    print('Creating model {}'.format(arg_model))
    # model.to(device)  # Transfer the model to the GPU/CPU

    # loss_function = nn.CrossEntropyLoss().cuda()
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters())  # Optimize all parameters

    return model, device, loss_function, optimizer
