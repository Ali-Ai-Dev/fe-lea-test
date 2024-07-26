import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Lambda


class FEMNIST_Train_Dataset(Dataset):
    def __init__(self, writer, binary_data_file, dic_train_indices, transform=None, target_transform=None):
        self.images = binary_data_file[writer]['images'][sorted(dic_train_indices[writer])]
        self.labels = binary_data_file[writer]['labels'][sorted(dic_train_indices[writer])]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            label = self.target_transform(label)
        
        return img, label


class FEMNIST_Test_Dataset(Dataset):
    def __init__(self, writers, binary_data_file, dic_test_indices, transform=None, target_transform=None):
        self.images = []
        self.labels = []
        for writer in writers:
            self.images.extend(binary_data_file[writer]['images'][sorted(dic_test_indices[writer])])
            self.labels.extend(binary_data_file[writer]['labels'][sorted(dic_test_indices[writer])])

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            label = self.target_transform(label)
        
        return img, label


def get_client_datasets(writers, binary_data_file, dic_train_indices):
    client_datasets = list()
    for writer in writers:
        client_dataset = FEMNIST_Train_Dataset(writer, binary_data_file, dic_train_indices,
                                               transform=ToTensor(),
                                               target_transform=Lambda(lambda y: torch.zeros(62)
                                                       .scatter_(dim=0, index=torch.tensor(y, dtype=torch.int64), value=1))
                                            )
        client_datasets.append(client_dataset)
    
    return client_datasets


def get_test_dataset(writers, binary_data_file, dic_test_indices):
    test_dataset = FEMNIST_Test_Dataset(writers, binary_data_file, dic_test_indices,
                                        transform=ToTensor(),
                                        target_transform=Lambda(lambda y: torch.zeros(62)
                                              .scatter_(dim=0, index=torch.tensor(y, dtype=torch.int64), value=1))
                                        )
    
    return test_dataset




