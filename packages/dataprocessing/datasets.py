import os
import random
import torch
import numpy as np
import glob
from PIL import Image
from torchvision import transforms as transforms_lib
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, SVHN
from torch.utils.data import random_split, Subset
from .segmentation_transforms import ToDtype, CenterCrop, PILToTensor, Normalize, RandomCrop, RandomHorizontalFlip, Compose
from ..utilities.logging_utilities import *
from tqdm import tqdm
from itertools import zip_longest


# Setup a logger
logger = setup_logger(__name__)

# **********************************************************************
# Description:
#   Class to handle custom image classification datasets.
# Parameters:
#   image_dir - Path to image directory
#   transform - Transformations to apply to the images
#   target_transform - Transformations to apply to the labels
#   pre_shuffle - Flag to shuffle the dataset upfront
# Notes:
#   Expected format of the the Custom Image Dataset follows 
#   Imagefolder format
#       root/class1/xxx.png
#       root/class1/xxy.png
#       root/class2/xxx.png
#       root/class2/xxy.png
# **********************************************************************
class CustomImageDataset(torch.utils.data.Dataset):
    # Initialize the transforms and the directory containing the images
    def __init__(self, image_dir, transform = None, target_transform = None, pre_shuffle = None):
        self.images = []
        self.labels = []
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform

        # Setup the dataset
        self.setup(pre_shuffle = pre_shuffle)

    # **********************************************************************
    # Description:
    #   Return the number of samples in the dataset. 
    # Parameters:
    #   -
    # Notes:
    #   -
    # **********************************************************************
    def __len__(self):
        return len(self.labels)

    # **********************************************************************
    # Description:
    #   Return a sample from the dataset at the given index(idx).
    # Parameters:
    #   idx - Index of sample to retrieve
    # Notes:
    #   -
    # **********************************************************************
    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    # **********************************************************************
    # Description:
    #   Get the image paths and their respective labels based on folder 
    #   structure.
    # Parameters:
    #   pre_shuffle - Flag to shuffle the dataset upfront
    # Notes:
    #   Pre-shuffle originally implemented for use during use of custom 
    #   samplers
    # **********************************************************************
    def setup(self, pre_shuffle = False):
        logger.debug('Setting up the dataset')
        list_of_classes = sorted(next(os.walk(self.image_dir))[1])

        for class_index, class_id in enumerate(list_of_classes):
            full_path = os.path.join(self.image_dir, class_id)

            images = sorted([os.path.join(full_path, file) for file in os.listdir(full_path)])
            
            num_of_examples = len(images)
            labels = [class_index] * num_of_examples

            self.images = self.images + images
            self.labels = self.labels + labels

        if pre_shuffle:
            logger.debug('Applying pre-shuffling to the dataset')
            # For post-process testing
            rand_idx = random.randint(0, len(self.labels) - 2)
            rand_img = self.images[rand_idx]

            shuffled_images_with_labels = []

            for (image, label) in zip(self.images, self.labels):
                shuffled_images_with_labels.append([image, label])
            
            random.shuffle(shuffled_images_with_labels)
            self.images = []
            self.labels = []
            for i, image_and_label in enumerate(shuffled_images_with_labels):
                self.images.append(image_and_label[0])
                self.labels.append(image_and_label[1])

            assert rand_img is not self.images[rand_idx], "Expected two images at same index pre and post-shuffle to not match, got match."

# Dataset: https://github.com/fastai/imagenette
class Imagenette(CustomImageDataset):
    def __init__(self, image_dir, pre_shuffle = False):
        
        transform = transforms_lib.Compose(
            [
                transforms_lib.Resize(256),
                transforms_lib.CenterCrop(224),
                transforms_lib.ToTensor(),
                transforms_lib.Normalize(
                    mean = (0.485, 0.456, 0.406), # Taken from torchvision models
                    std = (0.229, 0.224, 0.225), # Taken from torchvision models
                )
            ]
        )
        CustomImageDataset.__init__(self, image_dir, transform, None, pre_shuffle = pre_shuffle)

# Dataset: https://github.com/fastai/imagenette
class Imagewoof(CustomImageDataset):
    def __init__(self, image_dir, pre_shuffle = False):
        
        transform = transforms_lib.Compose(
            [
                transforms_lib.Resize(256),
                transforms_lib.CenterCrop(224),
                transforms_lib.ToTensor(),
                transforms_lib.Normalize(
                    mean = (0.485, 0.456, 0.406), # Taken from torchvision models
                    std = (0.229, 0.224, 0.225), # Taken from torchvision models
                )
            ]
        )
        CustomImageDataset.__init__(self, image_dir, transform, None, pre_shuffle = pre_shuffle)

# Dataset: https://www.image-net.org/
class Imagenet(CustomImageDataset):
    def __init__(self, image_dir, pre_shuffle = False):
        
        transform = transforms_lib.Compose(
            [
                transforms_lib.RandomResizedCrop(224),
                transforms_lib.RandomHorizontalFlip(p=0.5),
                transforms_lib.PILToTensor(),
                transforms_lib.ConvertImageDtype(torch.float),
                transforms_lib.Normalize(
                    mean = (0.485, 0.456, 0.406), # Taken from torchvision models
                    std = (0.229, 0.224, 0.225), # Taken from torchvision models
                )
            ]
        )
        CustomImageDataset.__init__(self, image_dir, transform, None, pre_shuffle = pre_shuffle)

# Dataset: https://www.image-net.org/
class ImagenetTest(CustomImageDataset):
    def __init__(self, image_dir, pre_shuffle = False):
        
        transform = transforms_lib.Compose(
            [
                transforms_lib.Resize(256),
                transforms_lib.CenterCrop(224),
                transforms_lib.ToTensor(),
                transforms_lib.Normalize(
                    mean = (0.485, 0.456, 0.406), # Taken from torchvision models
                    std = (0.229, 0.224, 0.225), # Taken from torchvision models
                )
            ]
        )
        CustomImageDataset.__init__(self, image_dir, transform, None, pre_shuffle = pre_shuffle)

# Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
# NOTE: This is gotten through torchvision.datasets
class CustomCIFAR10():
    def __init__(self, image_dir, pre_shuffle = False, data_augmentation = False):
        data_augmentation_transform = [
                # Horizontally flip the given image randomly with a given probability
                transforms_lib.RandomHorizontalFlip(p = 0.5),

                # Rotate the image by angle
                transforms_lib.RandomRotation(degrees = (-7, 7)),

                # Random affine transformation of the image keeping center invariant (*Rotation disabled)
                transforms_lib.RandomAffine(degrees = 0, shear = 10, scale = (0.8, 1.2)),

                # Randomly change the brightness, contrast, saturation and hue of an image.
                transforms_lib.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2)
            ]

        basic_transform = [
            transforms_lib.ToTensor(),
            transforms_lib.Normalize(
                mean = (0.49139968, 0.48215827 ,0.44653124), 
                std = (0.24703233, 0.24348505, 0.26158768)
            ),
        ]

        if data_augmentation:
            combined_transform = data_augmentation_transform + basic_transform
            self.transform = transforms_lib.Compose(combined_transform)
        else:
            self.transform = transforms_lib.Compose(basic_transform)
 
        self.dims = (3, 32, 32)
        self.image_dir = image_dir
        self.pre_shuffle = pre_shuffle

    def prepare_data(self):
        # download
        CIFAR10(self.image_dir, train=True, download=True)
        CIFAR10(self.image_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cifar10_full = CIFAR10(self.image_dir, train = True, download = True, transform = self.transform)
            dataset_size = len(cifar10_full)
            indicies = list(range(dataset_size))
            split = int(np.floor(0.1 * dataset_size))
            if self.pre_shuffle:
                np.random.seed(42)
                np.random.shuffle(indicies)
            train_indicies, val_indicies = indicies[split:], indicies[:split]
            
            self.train = Subset(cifar10_full, train_indicies)
            self.train.labels = cifar10_full.targets[split:]

            self.val = Subset(cifar10_full,val_indicies)
            self.val.labels = cifar10_full.targets[:split]
            
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = CIFAR10(self.image_dir, train=False, transform=self.transform)
            self.test.labels = self.test.targets

# Dataset: https://www.kaggle.com/datasets/zalando-research/fashionmnist
# NOTE: This is gotten through torchvision.datasets
class CustomFashionMNIST():
    def __init__(self, image_dir, pre_shuffle = False):
        self.transform = transforms_lib.Compose(
            [
                transforms_lib.ToTensor(),
                # TODO - These need checking
                transforms_lib.Lambda(lambda x: x.repeat(3, 1, 1)), # Change to color images
                transforms_lib.Normalize(
                    mean = (0.1307,), 
                    std = (0.3081,)
                ),
            ]
        )
 
        self.dims = (1, 28, 28)
        self.image_dir = image_dir
        self.pre_shuffle = pre_shuffle

    def prepare_data(self):
        # download
        FashionMNIST(self.image_dir, train=True, download=True)
        FashionMNIST(self.image_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            fashion_mnist_full = FashionMNIST(self.image_dir, train = True, download = True, transform = self.transform)
            dataset_size = len(fashion_mnist_full)
            indicies = list(range(dataset_size))
            split = int(np.floor(0.1 * dataset_size))
            if self.pre_shuffle:
                np.random.seed(42)
                np.random.shuffle(indicies)
            train_indicies, val_indicies = indicies[split:], indicies[:split]
            
            self.train = Subset(fashion_mnist_full, train_indicies)
            self.train.labels = fashion_mnist_full.targets[split:]

            self.val = Subset(fashion_mnist_full,val_indicies)
            self.val.labels = fashion_mnist_full.targets[:split]
            
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = FashionMNIST(self.image_dir, train=False, transform=self.transform)
            self.test.labels = self.test.targets

# Dataset: https://www.kaggle.com/datasets/hojjatk/mnist-dataset
# NOTE: This is gotten through torchvision.datasets
class CustomMNIST():
    def __init__(self, image_dir, pre_shuffle = False):
        self.transform = transforms_lib.Compose(
            [
                transforms_lib.ToTensor(),
                # TODO - These need checking
                transforms_lib.Lambda(lambda x: x.repeat(3, 1, 1)), # Change to color images
                transforms_lib.Normalize(
                    mean = (0.1307,), 
                    std = (0.3081,)
                ),
            ]
        )
 
        self.dims = (1, 28, 28)
        self.image_dir = image_dir
        self.pre_shuffle = pre_shuffle

    def prepare_data(self):
        # download
        MNIST(self.image_dir, train=True, download=True)
        MNIST(self.image_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.image_dir, train = True, download = True, transform = self.transform)
            dataset_size = len(mnist_full)
            indicies = list(range(dataset_size))
            split = int(np.floor(0.1 * dataset_size))
            if self.pre_shuffle:
                np.random.seed(42)
                np.random.shuffle(indicies)
            train_indicies, val_indicies = indicies[split:], indicies[:split]
            
            self.train = Subset(mnist_full, train_indicies)
            self.train.labels = mnist_full.targets[split:]

            self.val = Subset(mnist_full,val_indicies)
            self.val.labels = mnist_full.targets[:split]
            
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = MNIST(self.image_dir, train=False, transform=self.transform)
            self.test.labels = self.test.targets

# Dataset: http://ufldl.stanford.edu/housenumbers/
# NOTE: This is gotten through torchvision.datasets
class CustomSVHN():
    def __init__(self, image_dir, pre_shuffle = False):
        self.transform = transforms_lib.Compose(
            [
                transforms_lib.ToTensor(),
                # TODO - These need checking
                # transforms_lib.Lambda(lambda x: x.repeat(3, 1, 1)), # Change to color images
                transforms_lib.Normalize(
                    mean = (0.4376821, 0.4437697, 0.47280442), 
                    std = (0.19803012, 0.20101562, 0.19703614)
                ),
            ]
        )
 
        self.dims = (3, 32, 32)
        self.image_dir = image_dir
        self.pre_shuffle = pre_shuffle

    def prepare_data(self):
        # download
        SVHN(self.image_dir, train=True, download=True)
        SVHN(self.image_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            svhn_full = SVHN(self.image_dir, split = 'train', download = True, transform = self.transform)
            dataset_size = len(svhn_full)
            indicies = list(range(dataset_size))
            split = int(np.floor(0.1 * dataset_size))
            if self.pre_shuffle:
                np.random.seed(42)
                np.random.shuffle(indicies)
            train_indicies, val_indicies = indicies[split:], indicies[:split]
            
            self.train = Subset(svhn_full, train_indicies)
            self.train.labels = svhn_full.labels[split:]

            self.val = Subset(svhn_full,val_indicies)
            self.val.labels = svhn_full.labels[:split]
            
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = SVHN(self.image_dir, split='test', download = True, transform=self.transform)
            self.test.labels = self.test.labels
