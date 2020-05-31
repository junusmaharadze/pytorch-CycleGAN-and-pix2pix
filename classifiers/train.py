import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import time
import copy
import matplotlib.pyplot as plt
import numpy as np

from torchvision import models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from collections import defaultdict


class XbdDataLoader(object):
    """Custom data loader class for the xBD dataset. It requires the polygons and the labels file as input

    Attributes:
        data_dir (string): The path to the xBD_polygons_AB folder that contains subfolders A, B, and AB.
        Only B is needed here
        labels_file (string): The path to the labels textfile satellite_AB_labels.txt
        paths_labels_dict (dict): Dictionary that contains the true label for each image, taken from satellite_AB_labels.txt
        transforms (list): A list of transformations to apply to the data
    """

    def __init__(self, data_dir, labels_file, transforms):
        self.data_dir = data_dir
        self.labels_file = labels_file
        self.transforms = transforms
        self.paths_labels_dict = defaultdict(dict)
        self._construct_labels_dict()

    def _construct_labels_dict(self):
        with open(self.labels_file, 'r') as f:
            lines = [line.replace('\n', '') for line in f.readlines()]
            for line in lines:
                path, label = line.split(' ')
                split_path = path.split('/')
                filename = split_path[-1]
                split_name = split_path[-2]

                # Exclude labels for test
                if split_name in ['train', 'val']:
                    self.paths_labels_dict[split_name][filename] = int(label)

    def _get_images_and_labels(self, split_name):
        train_dir = os.path.join(self.data_dir, 'B', split_name)
        print('{} dir: {}'.format(split_name, train_dir))
        images = []
        labels = []

        for filename in os.listdir(train_dir):
            image_path = os.path.join(train_dir, filename)
            image = Image.open(image_path)
            transformed_image = data_transforms[split_name](image)
            images.append(transformed_image)

            # Get class label from labels_file corresponding to each image
            label = self.paths_labels_dict[split_name][filename]
            labels.append(label)

        return images, labels

    @property
    def train_data(self):
        images, labels = self._get_images_and_labels('train')
        return list(zip(images, labels))

    @property
    def val_data(self):
        images, labels = self._get_images_and_labels('val')
        return list(zip(images, labels))


def test_model(model, loss_function, data_loader):
    # set model in evaluation mode
    model.eval()

    current_loss = 0.0
    current_acc = 0

    # iterate over  the validation data
    for i, (inputs, labels) in enumerate(data_loader):
        # send the input/labels to the GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            loss = loss_function(outputs, labels)

        # statistics
        current_loss += loss.item() * inputs.size(0)
        current_acc += torch.sum(predictions == labels.data)

    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)

    print('Test Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss, total_acc))


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


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((128, 128)),
        # transforms.Rescale(142),
        # transforms.RandomCrip(128),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((128, 128)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


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
    parser.add_argument('--checkpoint_name', dest='checkpoint_name', type=str, default='resnet18_checkpoint',
                        help='The name of the best model\'s checkpoint to be used for inference')
    args = parser.parse_args()

    # for arg in vars(args):
    #     print('[%s] = ' % arg, getattr(args, arg))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)

    if args.model == 'resnet18':
        model = models.resnet18(pretrained=False, progress=True)
    # Add more models here

    print('Creating model {}'.format(args.model))
    model.to(device)  # Transfer the model to the GPU/CPU

    # Load the data
    data_loader = XbdDataLoader(data_dir=args.data_dir, labels_file=args.labels_file, transforms=data_transforms)

    train_data = data_loader.train_data
    train_loader = DataLoader(train_data, batch_size=16, num_workers=0, shuffle=True)
    val_data = data_loader.val_data
    val_loader = DataLoader(val_data, batch_size=16, num_workers=0, shuffle=True)

    dataloaders_dict = dict()
    dataloaders_dict['train'] = train_loader
    dataloaders_dict['val'] = val_loader

    # loss_function = nn.CrossEntropyLoss().cuda()
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters())  # Optimize all parameters

    # Train and eval the model
    best_val_model, train_stats, val_stats = train_val_model(model, dataloaders_dict, loss_function, optimizer, args.num_epochs)
    # Plots functionality is not yet completed
    # plot_curves(args.num_epochs, train_stats, val_stats)

    # Save best validation model
    save_checkpoint(best_val_model, args.checkpoint_name)

    # @TODO: Save best checkpoints
    # @TODO: Write test function
