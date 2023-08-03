
from torch.utils.data import DataLoader
import torch

import argparse

from model.solver import Solver
from dataloader.custom_dataloader import CustomDataset

def get_args():
    parser = argparse.ArgumentParser()   

    parser.add_argument('--model_name', type=str, default="first_train", help='name of the model to be saved/loaded')
    parser.add_argument('--annotations_file', type=str, default="./dataset/single/cropped_images.txt", help='name of the annotations file')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='number of elements in batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--opt', type=str, default='Adam', choices=['SGD', 'Adam'], help = 'optimizer used for training')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help = 'cpu used for training')
    parser.add_argument('--criterion', type=str, default='mse', choices=['mse', 'cross_entropy'], help = 'loss used for training')
    parser.add_argument('--writer_path', type=str, default = "./runs/experiments", help= "The path for Tensorboard metrics")
    parser.add_argument('--dataset_path', type=str, default='./dataset/single/', help='path were to save/get the dataset')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints', help='path where to save the trained model')
    parser.add_argument('--print_every', type=int, default=100, help='print losses every N iteration')
    parser.add_argument('--save_every', type=int, default=10, help='save model every N epochs')
    parser.add_argument('--resume_train', action='store_true', help='load the model from checkpoint before training')
    parser.add_argument('--scheduler', action='store_true', help='add scheduler during training')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'evaluate', 'debug'], help = 'net mode (train or test)')
    parser.add_argument('--manual_seed', type=bool, default=True, help='Use same random seed to get same train/valid/test sets for every training.')
    parser.add_argument('--use_tensorboard', type=bool, default=True, help='Use tensorboard during training.')

    return parser.parse_args()

def mean_std(loader):
    channel_means = []
    channel_stds = []
    
    for images, labels in loader:
        # Shape of images = [batch_size, channels, height, width]

        # Calculate the mean and std along the axis of the channels (axis=1)
        mean = images.mean(axis=(0, 2, 3))
        std = images.std(axis=(0, 2, 3))

        channel_means.append(mean)
        channel_stds.append(std)
        
    # Calculate the overall mean and std across all batches
    overall_mean = torch.stack(channel_means).mean(axis=0)
    overall_std = torch.stack(channel_stds).mean(axis=0)

    return overall_mean, overall_std

def main(args):
    BATCH_SIZE = args.batch_size # increase / decrease according to GPU memeory

    DEVICE = torch.device(args.device)
    if torch.cuda.is_available()==False and DEVICE=='cuda':
        DEVICE = torch.device("cpu")

    # use our dataset and defined transformations
    total_dataset = CustomDataset(args.annotations_file, args.dataset_path)
    print(len(total_dataset))

    total_len = len(total_dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    # split the dataset in train and test set
    if args.manual_seed:
        torch.manual_seed(1)
    indices = torch.randperm(len(total_dataset)).tolist()
    dataset = torch.utils.data.Subset(total_dataset, indices[:train_len])
    dataset_valid = torch.utils.data.Subset(total_dataset, indices[train_len : train_len + val_len])
    dataset_test = torch.utils.data.Subset(total_dataset, indices[train_len + val_len :])

    # define training and validation data loaders
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    data_loader_valid = DataLoader(
        dataset_valid, batch_size=BATCH_SIZE, shuffle=False)

    data_loader_test = DataLoader(
        dataset_test, batch_size=BATCH_SIZE, shuffle=False)

    print(len(dataset.indices))
    print(len(dataset_valid.indices))
    print(len(dataset_test.indices))

    print("Device: ", DEVICE)
    mean, std = mean_std(data_loader)
    print("mean and std: \n", mean, std)
    # define solver class
    solver = Solver(train_loader=data_loader,
            valid_loader=data_loader_valid,
            test_loader=data_loader_test,
            device=DEVICE,
            args=args,
            )

    # TRAIN model
    if args.mode == "train":
        solver.train()
    elif args.mode == "test":
        solver.test(img_count=50)
    elif args.mode == "evaluate":
        solver.evaluate(0)
    elif args.mode == "debug":
        solver.debug()
    else:
        raise ValueError("Not valid mode")

if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)
    
    