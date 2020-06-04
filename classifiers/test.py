import torch
import os
import time

from torch.utils.data import DataLoader
from tqdm import tqdm

from xBD_data_loader import XbdDataset
from utils import parse_arguments, initialize_model_and_params


def test_model(model, data_loader, loss_function, device):
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
    args = parse_arguments()
    model, device, loss_function, _ = initialize_model_and_params(args.model)

    checkpoint_path = os.path.join('./classifiers/checkpoints', args.checkpoint_name + '.pth')
    print('Loading model {} for inference'.format(checkpoint_path))

    model.load_state_dict(torch.load(checkpoint_path))  # Loading model for inference
    model.to(device)  # Transfer the model to the GPU/CPU

    # Load the test data
    test_dataset = XbdDataset(args.data_dir, args.labels_file, 'test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    # Test model on unseen images
    test_accuracy, test_loss = test_model(model, test_loader, loss_function, device)
