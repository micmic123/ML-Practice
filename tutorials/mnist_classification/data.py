import glob, os
from PIL import Image
from torch.utils.data import Dataset


class MNIST(Dataset):
    """
    A customized data loader for MNIST
    """
    def __init__(self, root, transform=None, preload=False):
        """
        Initialize the MNIST dataset
        :param
        - root: root dir of the dataset
        - transform: a custom transform function
        - preload: if preload the dataset into memory
        """
        self.images = None
        self.labels = None
        self.file_names = []
        self.root = root
        self.transform = transform
        self.len = 0

        # read file_names
        for i in range(10):
            file_paths = glob.glob(os.path.join(root, str(i), '*.png'))
            for file_path in file_paths:
                self.file_names.append((file_path, i))  # (file_path, label)

        # if preload dataset into memory
        if preload:
            self._preload()

        self.len = len(self.file_names)

    def _preload(self):
        """
        Preload dataset to memory
        """
        self.labels = []
        self.images = []

        for image_fn, label in self.file_names:
            # load images
            image = Image.open(image_fn)
            self.images.append(image.copy())
            # avoid too many opened files bug
            image.close()
            self.labels.append(label)

    # most important part to customize.
    def __getitem__(self, index):
        """
        Get a sample from the dataset
        """
        if self.images is not None:
            # If dataset is preloaded
            image = self.images[index]
            label = self.labels[index]
        else:
            # If on-demand data loading
            image_fn, label = self.file_names[index]
            image = Image.open(image_fn)

        # May use transform function to transform samples
        # e.g. random crop, whitening
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        """
        Total # of samples in the dataset
        """
        return self.len
