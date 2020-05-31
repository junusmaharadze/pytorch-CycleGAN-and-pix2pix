import os

from torchvision import transforms
from PIL import Image
from collections import defaultdict

normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# There is one transforms list for each train, val, test (although currently they are the same).
# Do NOT change the structure, in case we need to have different transformations for each split.
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((128, 128)),
        # transforms.Rescale(142),
        # transforms.RandomCrip(128),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        normalize_transform
    ]),
    'val': transforms.Compose([
        transforms.Resize((128, 128)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize_transform
    ]),
    'test': transforms.Compose([
        transforms.Resize((128, 128)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize_transform
    ])
}


class XbdDataLoader(object):
    """Custom data loader class for the xBD dataset. It requires the polygons and the labels file as input

    Attributes:
        data_dir (string): The path to the xBD_polygons_AB folder that contains subfolders A, B, and AB.
        Only B is needed here
        labels_file (string): The path to the labels textfile satellite_AB_labels.txt
        paths_labels_dict (dict): Dictionary that contains the true label for each image, taken from satellite_AB_labels.txt
    """

    def __init__(self, data_dir, labels_file):
        self.data_dir = data_dir
        self.labels_file = labels_file
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

                if split_name in ['train', 'val', 'test']:
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

    @property
    def test_data(self):
        images, labels = self._get_images_and_labels('test')
        return list(zip(images, labels))
