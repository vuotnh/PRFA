import os
import torch.utils.data as data
from torchvision.transforms import transforms
import cv2


class ImageDataset(data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.root_dir = root_dir
        self.image_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = f'{self.root_dir}\\{self.image_list[index]}'
        # image_path = f'{self.root_dir}/{self.image_list[index]}'
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 640))
        if self.transform is not None:
            image = self.transform(image)
        return image


def train_loader(root_dir, batch_size):
    agumentation = [
        transforms.ToPILImage(),
        # transforms.Resize(224),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    transform = transforms.Compose(agumentation)
    train_dataset = ImageDataset(root_dir, transform=transform)
    train_loader = data.DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
    return train_loader
