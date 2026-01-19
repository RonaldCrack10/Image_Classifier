from torchvision import datasets, transforms
import os
from torch.utils.data import DataLoader, random_split



cwd = os.getcwd()
folder_name = 'Dataset\cats_and_dogs'
folder_path = os.path.join(cwd, folder_name)
print(f"Folder path: {folder_path}")

# Data Loading and Transformation

cats_path = os.path.join(folder_path, 'cats')
dogs_path = os.path.join(folder_path, 'dogs')

data_tranform = transforms.Compose([
    transforms.Resize(size=(128, 128)),
    transforms.ToTensor()
])

# Loader creation

dataset = datasets.ImageFolder(root = folder_path, transform = data_tranform)
dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)

# Training and validation split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
test_loader = DataLoader(val_dataset, batch_size = 32, shuffle = True)


