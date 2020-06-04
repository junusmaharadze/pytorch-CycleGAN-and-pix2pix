import torch
import torch.nn as nn
import argparse
import os
import time

from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm

from xBD_data_loader import XbdDataset


def test_model(model, loss_function, data_loader):
    """Test best validation model on unseen images
    Args:
        model (torchvision.model): The model with the trained weights
        loss_function
        data_loader: Data loader for the test images

    Returns:
        total_acc (int), total_loss (int)
    """
    since = time.time()
    # set model in evaluation mode
    model.eval()

    current_loss = 0.0
    current_acc = 0

    print('data_loader', data_loader)
    # iterate over  the validation data
    for inputs, labels in tqdm(data_loader):
        # send input/labels to the GPU/CPU
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

    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Test Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss, total_acc))
    return total_acc, total_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test classifier')
    parser.add_argument('--model', dest='model', type=str, default='resnet18',
                        help='One of the models: resnet18|vgg4|densenet')
    parser.add_argument('--data_dir', dest='data_dir', type=str,
                        help='Path to the dataset containing the polygons and the labels textfile, eg: ./datasets/xBD_datasets/xBD_polygons_AB')
    parser.add_argument('--labels_file', dest='labels_file', type=str,
                        help='Path to the labels txt, eg: ./datasets/xBD_datasets/xBD_polygons_AB_csv/satellite_AB_labels.txt')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16)
    parser.add_argument('--checkpoint_name', dest='checkpoint_name', type=str, default='resnet18_checkpoint',
                        help='The name of the best model\'s checkpoint to be used for inference')
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=0)
    args = parser.parse_args()

    for arg in vars(args):
        print('[%s] =' % arg, getattr(args, arg))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    if args.model == 'resnet18':
        model = models.resnet18(pretrained=False, progress=True)
    # Add more models here

    checkpoint_path = os.path.join('./classifiers/checkpoints', args.checkpoint_name + '.pth')
    print('Loading model {} for inference'.format(checkpoint_path))

    model.load_state_dict(torch.load(checkpoint_path))  # Loading model for inference
    model.to(device)  # Transfer the model to the GPU/CPU

    # Load the test data
    test_dataset = XbdDataset(args.data_dir, args.labels_file, 'test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    # loss_function = nn.CrossEntropyLoss().cuda()
    loss_function = nn.CrossEntropyLoss().to(device)

    # Test model on unseen images
    test_accuracy, test_loss = test_model(model, loss_function, test_loader)
