import torch
import os
import time

from torch.utils.data import DataLoader
from tqdm import tqdm

from classifiers.xBD_data_loader import XbdDataset
from classifiers.utils import parse_arguments, initialize_model_and_params


def test_model(model, data_loader, loss_function, device, data_split_type='test'):
    """Test best validation model on unseen images
    Args:
        model (torchvision.model): The model with the trained weights
        loss_function
        data_loader: Data loader for the test images
        device: cpu or gpu
    Returns:
        total_acc (int), total_loss (int)
    """
    since = time.time()
    # set model in evaluation mode
    model.eval()

    current_loss = 0.0
    current_acc = 0

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
    print('Resnet18 Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('{:s} Set Resnet18 Test Loss: {:.4f}; Accuracy: {:.4f}'.format(data_split_type, total_loss, total_acc))
    return total_acc, total_loss


def main(**kwargs):
    model, device, loss_function, _ = initialize_model_and_params(kwargs['model'])

    checkpoint_path = os.path.join('./classifiers/checkpoints', kwargs['checkpoint_name'] + '.pth')
    print('Loading model {} for inference'.format(checkpoint_path))

    model.load_state_dict(torch.load(checkpoint_path))  # Loading model for inference
    model.to(device)  # Transfer the model to the GPU/CPU

    # Load the test data
    if 'pix2pix_interim' in kwargs:
        if kwargs['pix2pix_interim'] is True:
            test_dataset = XbdDataset(kwargs['data_dir'], kwargs['labels_file'], 'test', pix2pix_interim=True,
                                      current_img_paths=kwargs['current_img_paths'], current_labels=kwargs['current_labels'])
    else:
        test_dataset = XbdDataset(kwargs['data_dir'], kwargs['labels_file'], 'test')
    test_loader = DataLoader(test_dataset, batch_size=kwargs['batch_size'], num_workers=kwargs['num_workers'], shuffle=False)

    # Test model on unseen images
    test_accuracy, test_loss = test_model(model, test_loader, loss_function, device, kwargs['data_split_type'])


if __name__ == '__main__':
    args = parse_arguments()

    main(args)
