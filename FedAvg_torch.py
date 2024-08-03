# %%
# from google.colab import drive
# drive.mount('/content/drive')
# %cd "/content/drive/MyDrive/ColabTemp"

# %%
import torch
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import DataLoader
from torch import nn
from collections import OrderedDict
import matplotlib.pyplot as plt
import random
import copy
import os
import glob
import time
import numpy as np
import argparse
import shutil
import h5py
from FEMNIST_by_write import get_client_datasets, get_test_dataset

# %%
# Set seed for reproducing code
mySeed = 42
random.seed(mySeed)  # Python random module.
np.random.seed(mySeed)  # Numpy module.
torch.manual_seed(mySeed)
torch.cuda.manual_seed(mySeed)
torch.cuda.manual_seed_all(mySeed)  # if you are using multi-GPU.

dataloader_generator = torch.Generator()
dataloader_generator.manual_seed(mySeed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# %%
def float_range(mini, maxi):
    """Return function handle of an argument type function for 
       ArgumentParser checking a float range: mini <= arg <= maxi
         mini - minimum acceptable argument
         maxi - maximum acceptable argument"""

    # Define the function with default arguments
    def float_range_checker(arg):
        """New Type function for argparse - a float within predefined range."""

        try:
            f = float(arg)
        except ValueError:    
            raise argparse.ArgumentTypeError("must be a floating point number")
        if f < mini or f > maxi:
            raise argparse.ArgumentTypeError("must be in range [" + str(mini) + " .. " + str(maxi)+"]")
        return f

    # Return function handle to checking function
    return float_range_checker

# %%
# Hyper parameters (Auto)
parser = argparse.ArgumentParser(prog='FedSwap',
                                 description="FedSwap: Converging Federated Learning Faster",
                                 epilog="Written by Ali Bozorgzad based on FedSwap paper")

parser.add_argument("--dataset", "-d", dest="dataset_name", type=str, default="CIFAR10",
                    help="dataset name", choices=['MNIST', 'CIFAR10', 'CINIC10', 'FEMNIST', 'FEMNISTwriter'])
parser.add_argument("--NN_type", "-n", dest="neural_network_type", type=str, default="Conv2",
                    help="neural network type", choices=['MLP1', 'MLP2','Conv1','Conv2','Conv3', 'Conv4'])
parser.add_argument("--num_clients", "-c", dest="num_clients", type=int, default="10",
                    help="number of clients, except for 'FEMNISTwriter', cause it fixed")
parser.add_argument("--batch_size", "-b", dest="batch_size", type=int, default="64",
                    help="batch size")
parser.add_argument("--total_steps", "-t", dest="total_steps", type=int, default="301",
                    help="total steps")
parser.add_argument("--client_per", "-p", dest="client_select_percentage", type=float_range(1e-2, 1), default="1",
                    help="client selection percentage, between [1e-2...1] 1 is 100%%")
parser.add_argument("--clients_data_distribution", "-u", dest="clients_data_distribution", type=str, default="equal",
                    help="how distribute train data between clients", choices=['equal', 'random', 'normal'])
parser.add_argument("--random_split", "-r", dest="data_random_split", type=int, default="1",
                    help="data random split between clients", choices=[0, 1])
parser.add_argument("--learning_rate", "-l", dest="learning_rate", type=float, default="1e-3",
                    help="learning rate")
parser.add_argument("--client_epochs", "-e", dest="client_epochs", type=int, default="1",
                    help="client epochs")
parser.add_argument("--remain", dest="remain", type=float_range(1e-3, 1), default="1",
                    help="remain %% of dataset for running faster in test, between [1e-3...1] 1 is 100%%, except for 'FEMNISTwriter'")

args, unknown = parser.parse_known_args()


dataset_name = args.dataset_name
neural_network_type = args.neural_network_type

num_clients = args.num_clients
batch_size = args.batch_size
total_steps = args.total_steps
client_select_percentage = args.client_select_percentage
clients_data_distribution = args.clients_data_distribution
data_random_split = args.data_random_split

learning_rate = args.learning_rate
loss_fn = nn.CrossEntropyLoss()
client_epochs = args.client_epochs

remain = args.remain

# %%
# # Hyper parameters (Manual)
# dataset_name = "MNIST" # 'MNIST' or 'CIFAR10' or 'CINIC10' or 'FEMNIST' or 'FEMNISTwriter'
# neural_network_type = "Conv4" # 'MLP1' or 'MLP2' or'Conv1' or'Conv2' or'Conv3' or'Conv4'

# num_clients = 10 # except for 'FEMNISTwriter'
# batch_size = 16
# total_steps = 46
# client_select_percentage = 1
# clients_data_distribution = "normal" # 'equal' or 'random' or 'normal'
# data_random_split = 1 # 0 or 1

# learning_rate = 1e-4
# loss_fn = nn.CrossEntropyLoss()
# client_epochs = 1

# remain = 0.001 # Remove some data for running faster in test, except for 'FEMNISTwriter'

# %%
# Initialize parameters
client_selects = None
client_weights = None

passed_steps = 0

start_bold = "\u001b[1m"
end_bold = "\033[0m"
os.system("color")
color = {
    "ENDC": end_bold,
    "Bold": start_bold,
}

os.makedirs("datasets", exist_ok=True)
os.makedirs("save_log", exist_ok=True)

neural_network_classname = f"Neural_Network_{neural_network_type}"

if dataset_name == 'FEMNISTwriter':
    num_clients = 3597 # or 'num_clients = len(writers)' but put fstring after 'writers' is fill

save_file_name_pre = f"""FA
_{dataset_name}_{neural_network_type}
_{num_clients}c_{batch_size}b_{client_select_percentage}cp
_{clients_data_distribution}_{data_random_split}rs
_{learning_rate}lr_{client_epochs}ce_step"""
save_file_name_pre = save_file_name_pre.replace("\n", "")
print(f"save_log_file_name: '{save_file_name_pre}'")

# %%
# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using '{device}' device")

# %% [markdown]
# ## Load Data

# %%
def download_femnist_dataset_ready_to_read():
    ### ToDo: Download and extract dataset to datasets dir
    cur_dir = os.getcwd()
    datasets_dir = os.path.join(cur_dir, "datasets")

    if os.path.isdir(os.path.join(datasets_dir, "by_class")):
        dataset_dir = os.path.join(datasets_dir, "FEMNIST")
        os.rename(os.path.join(datasets_dir, "by_class"), dataset_dir)

        for i, class_dir in enumerate(os.listdir(dataset_dir)):
            class_imgs = os.path.join(dataset_dir, class_dir, "train_"+class_dir)
            shutil.move(class_imgs, dataset_dir)
            shutil.rmtree(os.path.join(dataset_dir, class_dir))
            print(f"Ready to be read and preprocess, class {i}.")


# %%
# Load dataset
if dataset_name == 'FEMNIST':
    download_femnist_dataset_ready_to_read()
    
    full_data = datasets.ImageFolder(
        'datasets/FEMNIST',
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(28, 28)),
            transforms.Grayscale(),
        ]),
        target_transform=Lambda(lambda y: torch.zeros(62).scatter_(dim=0, index=torch.tensor(y), value=1)),
    )

    lst_range = np.arange(0, len(full_data))
    lst_random = np.random.permutation(lst_range)
    test_indices = lst_random[: int(len(lst_random)*0.1)]
    train_indices = list(filter(lambda i: i not in test_indices, lst_range))

    train_data = torch.utils.data.Subset(full_data, train_indices)
    test_data = torch.utils.data.Subset(full_data, test_indices)

elif dataset_name == 'FEMNISTwriter':
    dataset_dir = "datasets\FEMNIST_by_write\write_all_labels.hdf5"
    binary_data_file = h5py.File(dataset_dir, "r")

    writers = sorted(binary_data_file.keys())
    dic_train_indices = dict()
    dic_test_indices = dict()
    len_train_data = 0

    for writer in writers:
        labels = binary_data_file[writer]['labels']

        lst_range = np.arange(0, len(labels))
        lst_random = np.random.permutation(lst_range)
        test_indices = lst_random[: int(len(lst_random)*0.1)]
        train_indices = list(filter(lambda i: i not in test_indices, lst_range))
        len_train_data += len(train_indices)

        dic_train_indices[writer] = train_indices
        dic_test_indices[writer] = test_indices

elif dataset_name == 'CINIC10':
    dataset_dir = 'datasets/CINIC-10'
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]

    train_data = datasets.ImageFolder(
        dataset_dir + '/train_valid',
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean, std=cinic_std),
        ]),
        target_transform=Lambda(lambda y: torch.zeros(10).scatter_(dim=0, index=torch.tensor(y), value=1)),
    )

    test_data = datasets.ImageFolder(
        dataset_dir + '/test',
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean, std=cinic_std),
        ]),
        target_transform=Lambda(lambda y: torch.zeros(10).scatter_(dim=0, index=torch.tensor(y), value=1)),
    )

else: # 'MNIST' or 'CIFAR10'
    running_dataset = getattr(datasets, dataset_name)

    train_data = running_dataset(
        root="datasets",
        train=True,
        download=True,
        transform=ToTensor(),
        target_transform=Lambda(lambda y: torch.zeros(10).scatter_(dim=0, index=torch.tensor(y), value=1)),    
    )

    test_data = running_dataset(
        root="datasets",
        train=False,
        download=True,
        transform=ToTensor(),
        target_transform=Lambda(lambda y: torch.zeros(10).scatter_(dim=0, index=torch.tensor(y), value=1)),
    )

if dataset_name != 'FEMNISTwriter':
    print(len(train_data))
    assert len(train_data) > (10*num_clients), ("we expect each client have some data")
    print(train_data[0][0].shape)
    print(train_data[0][1])

# %%
if dataset_name != 'FEMNISTwriter':
    # Remove some data for running faster in test
    print("remain data parameter:", remain)
    print("full train_data size:", len(train_data))
    train_data = torch.utils.data.Subset(train_data, range(0, int(len(train_data)*remain)))
    print("cutted train_data size:", len(train_data))

    print("full test_data size:", len(test_data))
    test_data = torch.utils.data.Subset(test_data, range(0, int(len(test_data)*remain)))
    print("cutted test_data size:", len(test_data))

# %%
def sum_one_normalized_random_numbers(size, mean=0, std=1):
    numbers = np.random.normal(loc=mean, scale=std, size=size)
    numbers = np.abs(numbers) # Ensure all numbers are positive
    numbers /= np.sum(numbers) # Normalize numbers to make their sum=1
    
    return numbers

# %%
def min_thresh_for_sum_one_random_numbers(sum_one_random_numbers):
    # set a min_thresh: ref='https://stackoverflow.com/a/62911965/4464934'
    min_thresh = 1 / len(train_data)
    rand_prop = 1 - num_clients * min_thresh
    random_numbers_min_thresh = (sum_one_random_numbers * rand_prop) + min_thresh

    return random_numbers_min_thresh

# %%
def client_data_size_for_sum_one_random_numbers(sum_one_random_numbers):
    random_numbers_min_thresh = min_thresh_for_sum_one_random_numbers(sum_one_random_numbers)
    client_data_size = np.floor(random_numbers_min_thresh*len(train_data)).astype(int)

    return client_data_size

# %%
# Split data to clients
if dataset_name != 'FEMNISTwriter':
    if clients_data_distribution == "equal":
        client_data_size = np.array([len(train_data)//num_clients]*num_clients)

    elif clients_data_distribution == "random":
        # random numbers with sum=1
        sum_one_random_numbers = np.random.dirichlet(np.ones(num_clients))
        client_data_size = client_data_size_for_sum_one_random_numbers(sum_one_random_numbers)
        
    elif clients_data_distribution == "normal":
        sum_one_random_numbers = sum_one_normalized_random_numbers(num_clients)
        client_data_size = client_data_size_for_sum_one_random_numbers(sum_one_random_numbers)

    data_remain = len(train_data) - sum(client_data_size)
    for i in range(data_remain):
        client_data_size[-1-i] += 1

    if data_random_split:
        client_datasets = torch.utils.data.random_split(train_data, client_data_size)
    else:
        client_datasets = list()
        i = 0
        for j in client_data_size:
            client_datasets.append(torch.utils.data.Subset(train_data, range(i, i+j)))
            i += j
else:
    client_datasets = get_client_datasets(writers, binary_data_file, dic_train_indices)
    test_data = get_test_dataset(writers, binary_data_file, dic_test_indices)

    print(f"num_clients in FEMNISTwriter: {len(client_datasets)}")
    print(len_train_data)
    print(client_datasets[0][0][0].shape)
    print(client_datasets[0][0][1])

# %%
# Create dataloader for each client
client_dataloaders = np.zeros(num_clients, dtype=object)
for i, dataset in enumerate(client_datasets):
    client_dataloaders[i] = DataLoader(dataset=dataset, batch_size=batch_size,
                                       shuffle=True, generator=dataloader_generator,)

test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size,
                             shuffle=True, generator=dataloader_generator,)

# %% [markdown]
# ## Training

# %%
def calc_out_conv_max_layers(in_w, in_h, kernels, strides, paddings=None, dilations=None):
    # In MaxPool2d, strides must same with kernels
    if paddings == None:
        paddings = np.zeros(len(kernels))
    
    if dilations == None:
        dilations = np.ones(len(kernels))
    
    out_w = in_w
    out_h = in_h
    for ker, pad, dil, stri in zip(kernels, paddings, dilations, strides):
        out_w = np.floor((out_w + 2*pad - dil * (ker-1) - 1)/stri + 1)
        out_h = np.floor((out_h + 2*pad - dil * (ker-1) - 1)/stri + 1)

    return int(out_w), int(out_h)

# %%
# Define MLP models
input_flat_size = torch.flatten(test_data[0][0]).shape[0]
nClasses = test_data[0][1].shape[0]

class Neural_Network_MLP1(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_flat_size, 100)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(100, 99)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(99, nClasses)),
        ]))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        probs = self.softmax(logits)
        return probs
    
    def get_weights(self):
        return list(self.parameters())
    
    def set_weights(self, parameters_list):
        for i, param in enumerate(self.parameters()):
            param.data = parameters_list[i].data


class Neural_Network_MLP2(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_flat_size, 256)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(256, 128)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(128, 64)),
            ('relu3', nn.ReLU()),
            ('fc4', nn.Linear(64, nClasses)),
        ]))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        probs = self.softmax(logits)
        return probs
    
    def get_weights(self):
        return list(self.parameters())
    
    def set_weights(self, parameters_list):
        for i, param in enumerate(self.parameters()):
            param.data = parameters_list[i].data

# %%
# Define Convolutional models
input_channels, input_width, input_height = test_data[0][0].shape

conv_kernel1 = 3
max_kernel1 = 2
kernels = [conv_kernel1, max_kernel1, conv_kernel1]
strides = [1, max_kernel1, 1]
out_w1, out_h1 = calc_out_conv_max_layers(input_width, input_height, kernels, strides)

class Neural_Network_Conv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.features_stack = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(input_channels, 32, kernel_size=conv_kernel1, stride=1, padding=0)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size=max_kernel1)),
            ('conv2', nn.Conv2d(32, 64, kernel_size=conv_kernel1)),
            ('relu2', nn.ReLU(inplace=True)),
            ('flat', nn.Flatten()),
            ('fc1', nn.Linear(64*out_w1*out_h1, nClasses)),
        ]))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.features_stack(x)
        probs = self.softmax(logits)
        return probs

    def get_weights(self):
        return list(self.parameters())
    
    def set_weights(self, parameters_list):
        for i, param in enumerate(self.parameters()):
            param.data = parameters_list[i].data


conv_kernel2 = 3
max_kernel2 = 2
kernels = [conv_kernel2, max_kernel2, conv_kernel2, max_kernel2]
strides = [1, max_kernel2, 1, max_kernel2]
out_w2, out_h2 = calc_out_conv_max_layers(input_width, input_height, kernels, strides)

class Neural_Network_Conv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.features_stack = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(input_channels, 32, kernel_size=conv_kernel2, stride=1, padding=0)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size=max_kernel2)),
            ('conv2', nn.Conv2d(32, 64, kernel_size=conv_kernel2)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(kernel_size=max_kernel2)),
            ('flat', nn.Flatten()),
            ('fc1', nn.Linear(64*out_w2*out_h2, 100)),
            ('relu3', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(100, nClasses)),
        ]))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.features_stack(x)
        probs = self.softmax(logits)
        return probs

    def get_weights(self):
        return list(self.parameters())
    
    def set_weights(self, parameters_list):
        for i, param in enumerate(self.parameters()):
            param.data = parameters_list[i].data


conv_kernel3 = 5
max_kernel3 = 2
kernels = [conv_kernel3, max_kernel3, conv_kernel3, max_kernel3]
strides = [1, max_kernel3, 1, max_kernel3]
paddings = [1, 1, 1, 1]
out_w3, out_h3 = calc_out_conv_max_layers(input_width, input_height, kernels, strides, paddings)

class Neural_Network_Conv3(nn.Module):
    def __init__(self):
        super().__init__()
        self.features_stack = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(input_channels, 32, kernel_size=conv_kernel3, stride=1, padding='same')),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size=max_kernel3)),
            ('conv2', nn.Conv2d(32, 64, kernel_size=conv_kernel3, padding='same')),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(kernel_size=max_kernel3)),
            ('flat', nn.Flatten()),
            ('fc1', nn.Linear(64*out_w3*out_h3, 2048)),
            ('relu3', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(2048, nClasses)),
        ]))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.features_stack(x)
        probs = self.softmax(logits)
        return probs

    def get_weights(self):
        return list(self.parameters())
    
    def set_weights(self, parameters_list):
        for i, param in enumerate(self.parameters()):
            param.data = parameters_list[i].data



# A Good model for MNIST dataset, Acc=99.6%
# https://medium.com/@BrendanArtley/mnist-keras-simple-cnn-99-6-731b624aee7f
# Layer order: Activation -> Normalization -> Pooling -> Dropout
conv_kernel4 = 3
max_kernel4 = 2
kernels = [conv_kernel4, conv_kernel4, max_kernel4, conv_kernel4, conv_kernel4, max_kernel4]
strides = [1, 1, max_kernel4, 1, 1, max_kernel4]
paddings = [1, 1, 0, 1, 1, 0]
out_w4, out_h4 = calc_out_conv_max_layers(input_width, input_height, kernels, strides, paddings)

class Neural_Network_Conv4(nn.Module):
    def __init__(self):
        super().__init__() 
        self.features_stack = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(input_channels, 32, kernel_size=conv_kernel4, stride=1, padding='same')),
            ('relu1', nn.ReLU(inplace=True)),
            ('bn1', nn.BatchNorm2d(32)),
            ('conv2', nn.Conv2d(32, 32, kernel_size=conv_kernel4, padding='same')),
            ('relu2', nn.ReLU(inplace=True)),
            ('bn2', nn.BatchNorm2d(32)),
            ('pool1', nn.MaxPool2d(kernel_size=max_kernel4)),
            ('drop1', nn.Dropout(p=0.25)),

            ('conv3', nn.Conv2d(32, 64, kernel_size=conv_kernel4, stride=1, padding='same')),
            ('relu3', nn.ReLU(inplace=True)),
            ('bn3', nn.BatchNorm2d(64)),
            ('conv4', nn.Conv2d(64, 64, kernel_size=conv_kernel4, padding='same')),
            ('relu4', nn.ReLU(inplace=True)),
            ('bn4', nn.BatchNorm2d(64)),
            ('pool2', nn.MaxPool2d(kernel_size=max_kernel4)),
            ('drop2', nn.Dropout(p=0.25)),

            ('flat', nn.Flatten()),
            ('fc1', nn.Linear(64*out_w4*out_h4, 512)),
            ('relu5', nn.ReLU(inplace=True)),
            ('bn5', nn.BatchNorm1d(512)),
            ('drop3', nn.Dropout(p=0.25)),

            ('fc2', nn.Linear(512, 1024)),
            ('relu6', nn.ReLU(inplace=True)),
            ('bn6', nn.BatchNorm1d(1024)),
            ('drop4', nn.Dropout(p=0.5)),

            ('fc3', nn.Linear(1024, nClasses)),
        ]))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.features_stack(x)
        probs = self.softmax(logits)
        return probs

    def get_weights(self):
        return list(self.parameters())
    
    def set_weights(self, parameters_list):
        for i, param in enumerate(self.parameters()):
            param.data = parameters_list[i].data

# %%
def select_clients_and_assign_weights(global_weights):
    global client_selects
    global client_weights

    lst = np.arange(0, num_clients)
    np.random.shuffle(lst)
    client_selects = lst[: int(len(lst)*client_select_percentage)]

    client_weights = {i: copy.deepcopy(global_weights)  for i in client_selects}

# %%
# Create an instantiate of a class with string value!
global_model = globals()[neural_network_classname]().to(device)
global_weights = global_model.get_weights()
select_clients_and_assign_weights(global_weights)
print(global_model)

global_history = {"times": {"train":list(), "step":list()},
                  "accuracy": list(),
                  "loss": list()}

# Load saved state & log
last_saved = sorted(glob.glob(f"save_log/{save_file_name_pre}_*.npz"), key=os.path.getmtime)
if last_saved:
    last_saved = last_saved[-1]
    passed_steps = int(last_saved.split("_")[-1].split(".")[0]) + 1

    npzFile = np.load(last_saved, allow_pickle=True)
    client_selects = npzFile["client_selects"]
    client_weights = npzFile["client_weights"].item()
    global_history = npzFile["global_history"].item()
    dataloader_generator_state = torch.tensor(npzFile["dataloader_generator_state"])
    random_state = tuple(npzFile["random_state_ndarray"])
    np_random_state = tuple(npzFile["np_random_state_ndarray"])
    # torch_rng_states = npzFile["torch_rng_states_ndarray"]
    npzFile.close()

    dataloader_generator.set_state(dataloader_generator_state)
    random.setstate(random_state)
    np.random.set_state(np_random_state)
    # torch.set_rng_state(torch_rng_states[0])
    # torch.cuda.set_rng_state(torch_rng_states[1])
    # torch.cuda.set_rng_state_all(torch_rng_states[2])

# %%
def scale_model_weights(weights, scalar):
    """ Scale the model weights """

    scaled_weights = list()
    for i in range(len(weights)):
        scaled_weights.append(weights[i] * scalar)

    return scaled_weights

# %%
def sum_scaled_weights(client_scaled_weights):
    """ Return the sum of the listed scaled weights.
        axis_O is equivalent to the average weight of the weights """

    avg_weights = list()
    # get the average gradient accross all client gradients
    for gradient_list_tuple in zip(*client_scaled_weights):
        gradient_list_tuple = [tensors.tolist()  for tensors in gradient_list_tuple]
        layer_mean = torch.sum(torch.tensor(gradient_list_tuple), axis=0).to(device)
        avg_weights.append(layer_mean)

    return avg_weights


### Explaining the function with example ###
# t = (torch.tensor([[[2, 3],[3, 4]], [[3, 4],[4, 5]], [[4, 5],[5, 6]]]),
#      torch.tensor([[[5, 6],[6, 7]], [[6, 7],[7, 8]], [[7, 8],[8, 9]]]))
# t = [i.tolist() for i in t]
# for y in zip(*t):
#     print(y)
#     print(torch.sum(torch.tensor(y), axis=0))

# %%
def fed_avg():
    # calculate total training data across clients
    global_count = 0
    for client in client_selects:
        global_count += len(client_dataloaders[client].dataset)

    # initial list to collect clients weight after scalling
    client_scaled_weights = list()
    for client in client_selects:
        local_count = len(client_dataloaders[client].dataset)
        scaling_factor = local_count / global_count
        scaled_weights = scale_model_weights(client_weights[client], scaling_factor)
        client_scaled_weights.append(scaled_weights)

    # to get the average over all the clients model, we simply take the sum of the scaled weights
    avg_weights = sum_scaled_weights(client_scaled_weights)

    return avg_weights

# %%
def test_neural_network(dataloader, model, loss_fn):
    data_size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct_items = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct_items += (pred.argmax(1) == y.argmax(1)).sum().item()

    avg_loss = test_loss / num_batches
    accuracy = correct_items / data_size
    # print(f"Test Error: \nAccuracy: {(accuracy*100):>0.1f}%, Loss: {avg_loss:>8f}\n")

    return accuracy, avg_loss

# %%
def train_neural_network(dataloader, model, loss_fn, optimizer):
    data_size = len(dataloader.dataset)
    running_loss = 0

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        print_step = np.ceil(len(dataloader)/10)
        if batch % print_step == 0:
            loss_per_batch = running_loss / print_step
            current_item = (batch+1)*len(x)
            # print(f"loss: {loss_per_batch:>7f}  [{current_item:>5d}/{data_size:>5d}]")
            running_loss = 0

# %%
def train_clinet(dataloader, model):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    for epoch in range(client_epochs):
        train_neural_network(dataloader, model, loss_fn, optimizer)

# %%
def save_state_and_log(step):
    # save torch_rng, if you need that
    # torch_rng_states = [torch.get_rng_state(), torch.cuda.get_rng_state(), torch.cuda.get_rng_state_all()]
    # torch_rng_states_ndarray = np.array(torch_rng_states, dtype=object)

    dataloader_generator_state = dataloader_generator.get_state()
    random_state = random.getstate()
    np_random_state = np.random.get_state()
    random_state_ndarray = np.array(random_state, dtype=object)
    np_random_state_ndarray = np.array(np_random_state, dtype=object)

    np.savez_compressed(f"save_log/{save_file_name_pre}_{step}.npz",
                        client_selects=client_selects,
                        client_weights=client_weights,
                        global_history=global_history,
                        dataloader_generator_state=dataloader_generator_state,
                        random_state_ndarray=random_state_ndarray,
                        np_random_state_ndarray=np_random_state_ndarray,
                        # torch_rng_states_ndarray=torch_rng_states_ndarray,
                        )

    if step != 0:
        os.remove(f"save_log/{save_file_name_pre}_{step-1}.npz")

# %%
def change_time_format(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    if h:
        return f"{h:.0f}h-{m:.0f}m-{s:.0f}s"
    elif m:
        return f"{m:.0f}m-{s:.0f}s"
    else:
        return f"{s:.2f}s"

# %%
def print_log(training_time, step_time, step, metric_index=-1):
    training_time = change_time_format(training_time)
    step_time = change_time_format(step_time)
    print(f"round: {step} | training_time: {training_time} | step_time: {step_time}")
    print(f"round: {step} / global_acc: {start_bold}{global_history['accuracy'][metric_index]:.4%}{end_bold} / global_loss: {start_bold}{global_history['loss'][metric_index]:.4f}{end_bold}\n")

# %%
def print_prev_log(passed_steps):
    if passed_steps:
        for step in range(passed_steps):
            training_time = global_history["times"]["train"][step]
            step_time = global_history["times"]["step"][step]
            print_log(training_time, step_time, step)

# %%
# FedAvg Main Loop
print_prev_log(passed_steps)
for step in range(passed_steps, total_steps):
    training_time_start = time.time()
    for client in client_selects:
        local_model = globals()[neural_network_classname]().to(device)
        local_model.set_weights(client_weights[client])
        train_clinet(client_dataloaders[client], local_model)
        client_weights[client] = local_model.get_weights()

        del local_model
    
    training_time = time.time() - training_time_start
    global_history["times"]["train"].append(training_time)


    avg_weights = fed_avg()
    global_model.set_weights(avg_weights) # update global model
    select_clients_and_assign_weights(avg_weights)
    global_acc, global_loss = test_neural_network(test_dataloader, global_model, loss_fn)
    global_history["accuracy"].append(global_acc)
    global_history["loss"].append(global_loss)
    
    
    step_time = time.time() - training_time_start
    global_history["times"]["step"].append(step_time)
    print_log(training_time, step_time, step)
    save_state_and_log(step)

# %% [markdown]
# ## Result

# %%
# plt.plot(global_history["loss"], label="test loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Test Data")
# plt.legend()
# plt.show()

# %%
# plt.plot(global_history["accuracy"], label="test accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.title("Test Data")
# plt.legend()
# plt.show()

# %%
# plt.plot(global_history["loss"], label="test loss")
# plt.plot(global_history["accuracy"], label="test accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Loss / Accuracy")
# plt.title("Test Data")
# plt.legend()
# plt.show()

# %%



