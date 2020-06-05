import os

from torch.utils.data import Dataset
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


class XbdDataset(Dataset):
    def __init__(self, data_dir, labels_file, data_split, pix2pix_interim=False,
                 current_img_paths=None, current_labels=None):
        self.data_dir = data_dir
        self.labels_file = labels_file
        self.data_split = data_split
        self.paths_labels_dict = defaultdict(dict)
        if pix2pix_interim is False:
            self._construct_labels_dict()
        else:
            self.pix2pix_interim = pix2pix_interim
            self.filenames = current_img_paths
            self.labels = [int(x) for x in current_labels]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        # Generates one sample of the data
        if self.pix2pix_interim is False:
            images_dir = os.path.join(self.data_dir, 'B', self.data_split)
            image_path = os.path.join(images_dir, self.filenames[index])
        else:
            images_dir = self.data_dir
            image_path = self.filenames[index]

        image = Image.open(image_path)
        transformed_image = data_transforms[self.data_split](image)
        label = self.labels[index]

        return transformed_image, label

    def _construct_labels_dict(self):
        with open(self.labels_file, 'r') as f:
            lines = [line.replace('\n', '') for line in f.readlines()]
            filenames = []
            labels = []

            for line in lines:
                path, label = line.split(' ')
                split_path = path.split('/')
                filename = split_path[-1]

                if self.data_split in path:
                    self.paths_labels_dict[filename] = int(label)
                    filenames.append(filename)
                    labels.append(int(label))
        self.filenames = filenames
        self.labels = labels
        print('{} dir size: {}'.format(self.data_split, len(self.filenames)))
