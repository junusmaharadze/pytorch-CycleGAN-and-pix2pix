import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import time
import copy
import matplotlib.pyplot as plt
import numpy as np

from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm

from xBD_data_loader import XbdDataLoader


def train_val_model(model, dataloaders, criterion, optimizer, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_stats = {'accuracy': [], 'loss': []}
    val_stats = {'accuracy': [], 'loss': []}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to eval mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'train':
                train_stats['accuracy'].append(epoch_acc)
                train_stats['loss'].append(epoch_loss)
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_stats['accuracy'].append(epoch_acc)
                val_stats['loss'].append(epoch_loss)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_stats, val_stats


def plot_curves(num_epochs, train_stats, val_stats):
    """Plots accuracy and loss curves for training and validation
    Args:
        num_epochs (int): Number of epochs the model ran for
        train_stats (dict): Accuracy and loss stats for training
        val_stats (dict): Accuracy and loss stats for validation
    """
    for metric in ['accuracy', 'loss']:
        train_stats[metric]
        val_stats[metric]

        plt.title('Training/Validation {} vs. Number of Training Epochs'.format(metric))
        plt.xlabel('Training Epochs')
        plt.ylabel(metric)
        plt.plot(range(1, num_epochs + 1), train_stats[metric], label='Training')
        plt.plot(range(1, num_epochs + 1), val_stats[metric], label='Validation')
        plt.xticks(np.arange(1, num_epochs + 1, 1.0))
        plt.ylim((0, 1.))
        plt.legend()
        plt.show()


def save_checkpoint(model, checkpoint):
    """Saves best validation checkpoint in checkpoints_dir
        Args:
        model (torchvision.model): Best validation model's weights
        checkpoint (string): Name of the checkpoint file
    """
    checkpoints_dir = './classifiers/checkpoints'
    if not os.path.isdir(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    checkpoint_path = os.path.join(checkpoints_dir, checkpoint + '.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print('Saved checkpoint: {}'.format(checkpoint_path))


if __name__ == '__main__':
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

    # for arg in vars(args):
    #     print('[%s] = ' % arg, getattr(args, arg))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    if args.model == 'resnet18':
        model = models.resnet18(pretrained=False, progress=True)
    # Add more models here

    print('Creating model {}'.format(args.model))
    model.to(device)  # Transfer the model to the GPU/CPU

    # Load the data
    data_loader = XbdDataLoader(data_dir=args.data_dir, labels_file=args.labels_file)

    train_data = data_loader.train_data
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_data = data_loader.val_data
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    dataloaders_dict = dict()
    dataloaders_dict['train'] = train_loader
    dataloaders_dict['val'] = val_loader

    # loss_function = nn.CrossEntropyLoss().cuda()
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters())  # Optimize all parameters

    # Train and eval the model
    best_val_model, train_stats, val_stats = train_val_model(model, dataloaders_dict, loss_function, optimizer, args.num_epochs)
    # Plots functionality is not yet completed
    plot_curves(args.num_epochs, train_stats, val_stats)

    # Save best validation model
    save_checkpoint(best_val_model, args.checkpoint_name)
