import os
import cv2
import torch
from torch.utils.data import Dataset

from PIL import Image
import torchvision.transforms  as T

def get_train_transform(image):
    transf = T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        T.Normalize(mean = [0.6227], 
                         std= [0.3223])
    ])
    return transf(image)
def normalize_labels(labels):
    transf = T.Normalize(mean= [15.9816], std=[0.3119])
    return transf(labels)

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('L')
        return img

class CustomDataset(Dataset):
    def __init__(self, data_file, data_dir):
        self.data_file = data_file
        self.data_dir = data_dir

        # Read the data from the text file and store it as a list of tuples (image_name, x_label, y_label)
        with open(self.data_file, "r") as file:
            self.data = [line.strip().split() for line in file]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, x_label, y_label = self.data[idx]

        # Load the image
        image_path = os.path.join(self.data_dir, image_name)
        image = pil_loader(image_path)
        # Your image preprocessing code goes here (e.g., resizing, normalization)
        # Your image preprocessing code goes here (e.g., resizing, normalization)
        image = get_train_transform(image)
        # Convert x and y labels to PyTorch tensors
        label_tensor = torch.tensor([float(x_label),float(y_label)], dtype=torch.float32)
        #label_tensor = (label_tensor - 15.9816)/0.3119
        return image, label_tensor
