import torch
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from torchvision import datasets, transforms
import config


class DataPipeline:

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
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])


    def __dataset_loader(self, path):

        # train dataset element processing
        train_set = datasets.ImageFolder(
            root=path,
            transform=self.train_transform
        )

        # validation dataset element processing
        validation_set = datasets.ImageFolder(
            root=path,
            transform=self.validation_test_transform
        )

        # test dataset element processing
        test_set = datasets.ImageFolder(
            root=path,
            transform=self.validation_test_transform
        )

        # fixed random seed, every run produces the same train/validation/test split
        generator = torch.Generator().manual_seed(config.seed)

        # splitting the dataset according to its indexes
        train_indices, validation_indices, test_indices = random_split(
            train_set,
            lengths = [config.train_size, config.validation_size, config.test_size],
            generator = generator
        )

        train_subset = Subset(
            train_set,
            indices = train_indices.indices
        )

        validation_subset = Subset(
            validation_set,
            indices = validation_indices.indices
        )

        test_subset = Subset(
            test_set,
            indices=test_indices.indices
        )

        return train_subset, validation_subset, test_subset

    def __data_combiner(self):

        # loading different datasets
        buffalo_train, buffalo_validation, buffalo_test = self.__dataset_loader(config.buffalo)
        elephant_train, elephant_validation, elephant_test = self.__dataset_loader(config.elephant)
        rhino_train, rhino_validation, rhino_test = self.__dataset_loader(config.rhino)
        zebra_train, zebra_validation, zebra_test = self.__dataset_loader(config.zebra)

        train_data = ConcatDataset(
            [
                buffalo_train,
                elephant_train,
                rhino_train,
                zebra_train,
            ]
        )

        validation_data = ConcatDataset(
            [
                buffalo_validation,
                elephant_validation,
                rhino_validation,
                zebra_validation,
            ]
        )

        test_data = ConcatDataset(
            [
                buffalo_test,
                elephant_test,
                rhino_test,
                zebra_test,
            ]
        )

        return train_data, validation_data, test_data