# Plots evaluation metrics (accuracy, loss, f1 score) for the gan_eval csv files
# generated during the GAN training, where generated images are being evaluated
# by the classifier.

import argparse
import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import defaultdict


def plot_curves(num_epochs, train_stats, val_stats):
    """Plots accuracy and loss curves for training and validation
    Args:
        num_epochs (int): Number of epochs the model ran for
        train_stats (dict): Accuracy and loss stats for training
        val_stats (dict): Accuracy and loss stats for validation
    """
    for metric in ['accuracy', 'f1', 'loss']:
        plt.title('Training/Validation {}'.format(metric))
        plt.xlabel('Training Epochs')
        plt.ylabel(metric)
        plt.plot(range(1, num_epochs + 1), train_stats[metric], label='Training')
        plt.plot(range(1, num_epochs + 1), val_stats[metric], label='Validation')
        plt.xticks(np.arange(1, num_epochs + 1, 1.0))
        # plt.ylim((0, 1.))
        plt.legend()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train classifier')
    parser.add_argument('--csv_file', dest='csv_file', type=str,
                        help='Path to the csv file containing the train/val metrics')
    args = parser.parse_args()

    columns = defaultdict(lambda: defaultdict(list))

    with open(args.csv_file, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        headers = [r.strip() for r in next(reader)]

        for j, row in enumerate(reader):
            split_type = row[0]

            for i in range(2, len(headers)):
                metric_name = headers[i]
                metric_name = metric_name.split('test_')[1]
                columns[split_type][metric_name].append(float(row[i]))

    columns = dict(columns)
    num_epochs = len(columns['train']['accuracy'])
    plot_curves(num_epochs, columns['train'], columns['val'])
