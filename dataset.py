import torch
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from torchvision import datasets, transforms
import config


class RelabeledDataset(torch.utils.data.Dataset):
    def __init__(self, subset, new_label):
        self.subset = subset
        self.new_label = new_label

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, _ = self.subset[idx]
        return image, self.new_label


class DataPipeline:

    def __init__(self):

        # transform train dataset elements
        self.train_transform = transforms.Compose([

            # resize the image
            transforms.Resize(
                (config.image_size, config.image_size)
            ),

            # to avoid underfitting or overfitting I am going to flip half of dataset elements
            transforms.RandomHorizontalFlip(p=0.5),

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


    def __dataset_loader(self, path, label):

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

        train_subset = RelabeledDataset(
            Subset(
                train_set,
                indices=train_indices.indices
            ),
            label
        )

        validation_subset = RelabeledDataset(
            Subset(
                validation_set,
                indices=validation_indices.indices
            ),
            label
        )
        test_subset = RelabeledDataset(
            Subset(
                test_set,
                indices=test_indices.indices),
            label
        )

        return train_subset, validation_subset, test_subset

    def __data_combiner(self):

        # loading different datasets
        buffalo_train, buffalo_validation, buffalo_test = self.__dataset_loader(config.buffalo, label=0)
        elephant_train, elephant_validation, elephant_test = self.__dataset_loader(config.elephant, label=1)
        rhino_train, rhino_validation, rhino_test = self.__dataset_loader(config.rhino, label=2)
        zebra_train, zebra_validation, zebra_test = self.__dataset_loader(config.zebra, label=3)

        combined_train_data = ConcatDataset(
            [
                buffalo_train,
                elephant_train,
                rhino_train,
                zebra_train,
            ]
        )

        combined_validation_data = ConcatDataset(
            [
                buffalo_validation,
                elephant_validation,
                rhino_validation,
                zebra_validation,
            ]
        )

        combined_test_data = ConcatDataset(
            [
                buffalo_test,
                elephant_test,
                rhino_test,
                zebra_test,
            ]
        )

        return combined_train_data, combined_validation_data, combined_test_data

    def load_dataset(self):
        train_data, validation_data, test_data = self.__data_combiner()

        train_loader = DataLoader(
            train_data,
            batch_size = config.batch_size,
            shuffle = True,
            num_workers=config.num_workers
        )

        validation_loader = DataLoader(
            validation_data,
            batch_size = config.batch_size,
            shuffle = False,
            num_workers=config.num_workers
        )

        test_loader = DataLoader(
            test_data,
            batch_size = config.batch_size,
            shuffle = False,
            num_workers=config.num_workers
        )

        return train_loader, validation_loader, test_loader