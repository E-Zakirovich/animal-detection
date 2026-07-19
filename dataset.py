import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import config


class Dataset:

    def __init__(self):

        # transform train dataset elements
        self.train_transform = transforms.Compose([

            # resize the image
            transforms.Resize(
                (config.image_size, config.image_size)
            ),

            # to avoid underfitting or overfitting I am going to flip half of dataset elements
            transforms.RandomVerticalFlip(p=0.5),

            # main reason for rotate the image is, avoid underfitting or overfitting
            transforms.RandomRotation(degrees=10),

            # to teach my cnn, I need to change dataset elements to tensors
            transforms.ToTensor(),

            # normalization
            transforms.Normalize(
                mean = (0.485, 0.456, 0.406),
                std = (0.229, 0.224, 0.225)
            )
        ])

        # transform test & train dataset elements
        self.validation_test_transform = transforms.Compose([

            # resize the image
            transforms.Resize(
                (config.image_size, config.image_size)
            ),

            # to teach my cnn, I need to change dataset elements to tensors
            transforms.ToTensor(),

            # normalization part
            transforms.Normalize()
        ])


    def load_buffalo(self):

        # train dataset element processing
        train_set = datasets.ImageFolder(
            root=config.buffalo,
            transform=self.train_transform
        )

        # validation dataset element processing
        validation_set = datasets.ImageFolder(
            root=config.buffalo,
            transform=self.validation_test_transform
        )

        # test dataset element processing
        test_set = datasets.ImageFolder(
            root=config.buffalo,
            transform=self.validation_test_transform
        )
        

    def load_elephant(self):
        ...

    def load_rhino(self):
        ...

    def load_zebra(self):
        ...

    def load_dataset(self):
        ...