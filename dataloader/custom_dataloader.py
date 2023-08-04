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
         T.Normalize(mean = [0.5507], 
            std= [0.2963])                 
    ])
    return transf(image)

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

        with open(self.data_file, "r") as file:
            self.data = [line.strip().split() for line in file]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, x_label, y_label = self.data[idx]

        image_path = os.path.join(self.data_dir, image_name)
        image = pil_loader(image_path)
        image = get_train_transform(image)
        label_tensor = torch.tensor([float(x_label),float(y_label)], dtype=torch.float32)
        return image, label_tensor
