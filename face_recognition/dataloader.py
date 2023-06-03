import os
import torchvision.transforms.functional as F
import torch
import random
import math
import uuid
from PIL import Image
import shutil
from torchvision import transforms
from torch.utils.data import ConcatDataset, DataLoader, SubsetRandomSampler, Dataset



def remove_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Data augmentation
def augmentation(img):
    data = []

    img = F.adjust_brightness(img, brightness_factor=1.05)
    data.append(img)

    img = F.adjust_contrast(img, contrast_factor=torch.empty(1).uniform_(0.6, 1).item())
    data.append(img)

    img = F.hflip(img)
    data.append(img)

    img = F.adjust_saturation(img, saturation_factor=torch.empty(1).uniform_(0.9, 1).item())
    data.append(img)

    return data


# select index of images for split to anchors and positives
def indices_split_data(pool_address):
    pool_images = os.listdir(pool_address)
    indices = list(range(len(pool_images)))
    random.shuffle(indices)
    split = int(math.floor(0.8 * len(pool_images)))
    anchor_indices, positive_indices = indices[split:], indices[:split]
    anchor_images = [pool_images[i] for i in anchor_indices]
    positive_images = [pool_images[i] for i in positive_indices]
    return anchor_images, positive_images


def split(root_path, pool_address, pos_path, anc_path):
    anchor_images, positive_images = indices_split_data(pool_address)

    for image_name in anchor_images:
        source_path = os.path.join(pool_address, image_name)
        destination_path = os.path.join(anc_path)
        shutil.copy2(source_path, destination_path)

    for image_name in positive_images:
        source_path = os.path.join(pool_address, image_name)
        destination_path = os.path.join(pos_path)
        shutil.copy2(source_path, destination_path)


# processing data augmentation
def processing_data_augmentation(type, pos_path, anc_path):
    path = pos_path if type == "positive" else anc_path
    files = os.listdir(os.path.join(path))
    count = 0
    while count < ((2500 // (len(files) * 4)) - 1):
        for file_name in files:
            img_path = os.path.join(path, file_name)
            img = Image.open(img_path).convert('RGB')
            img_tensor = F.to_tensor(img)
            augmented_images = augmentation(img_tensor)
            for i, image in enumerate(augmented_images):
                image = F.to_pil_image(image)
                image.save(os.path.join(path, '{}.jpg'.format(uuid.uuid1())))

        count = count + 1


    while 2499 >= len(files):
        files = os.listdir(os.path.join(path))
        rnd = random.randint(0, len(files))
        file_name = files[rnd]
        img_path = os.path.join(path, file_name)
        img = Image.open(img_path).convert('RGB')
        img.save(os.path.join(path, '{}.jpg'.format(uuid.uuid1())))
        files = os.listdir(os.path.join(path))


def delete_folder(mypath):
    try:
        shutil.rmtree(mypath)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

# Create custom datasets

class MergeImageDataset(Dataset):
    """
    (anchor, negative, 0) -> types = 0
    (anchor, positive, 1) -> types = 1
    """

    def __init__(self, ANC_PATH, POS_PATH, NEG_PATH, types, transform=None):

        self.POS_PATH = POS_PATH
        self.NEG_PATH = NEG_PATH
        self.ANC_PATH = ANC_PATH

        self.types = types

        self.transform = transform

        self.POS_IMG = os.listdir(POS_PATH)
        self.NEG_IMG = os.listdir(NEG_PATH)
        self.ANC_IMG = os.listdir(ANC_PATH)

    def __len__(self):
        return len(self.ANC_IMG)

    def __getitem__(self, idx):

        anc_dir = os.path.join(self.ANC_PATH, self.ANC_IMG[idx])
        anc_image = Image.open(anc_dir).convert('RGB')

        if self.types == 1:
            pos_dir = os.path.join(self.POS_PATH, self.POS_IMG[idx])
            pos_image = Image.open(pos_dir).convert('RGB')

        if self.types == 0:
            neg_dir = os.path.join(self.NEG_PATH, self.NEG_IMG[idx])
            neg_image = Image.open(neg_dir).convert('RGB')

        data = [anc_image, pos_image if self.types == 1 else neg_image,
                torch.ones(1) if self.types == 1 else torch.zeros(1)]

        if self.transform:
            data[0] = self.transform(data[0])
            data[1] = self.transform(data[1])

        return data


class TestDataset(Dataset):
    """
    (input, positive, 0) -> not similar
    (input, positive, 1) -> similar
    """

    def __init__(self, INPUT_PATH, POS_PATH, transform=None):
        self.POS_PATH = POS_PATH
        self.INPUT_PATH = INPUT_PATH

        self.transform = transform

        self.POS_IMG = os.listdir(POS_PATH)
        self.INPUT_IMG = os.listdir(INPUT_PATH)

    def __len__(self):
        return len(self.INPUT_IMG)

    def __getitem__(self, idx):
        input_dir = os.path.join(self.INPUT_PATH, self.INPUT_IMG[idx])
        input_image = Image.open(input_dir).convert('RGB')

        pos_dir = os.path.join(self.POS_PATH, self.POS_IMG[idx])
        pos_image = Image.open(pos_dir).convert('RGB')

        data = [input_image, pos_image]

        if self.transform:
            data[0] = self.transform(data[0])
            data[1] = self.transform(data[1])

        return data

    def GetFileName(self):
        return self.INPUT_IMG


# package whole things' dataset + data loader + processing + split
def get_train_data(anc_path, pos_path, neg_path):
    transform = transforms.Compose([transforms.Resize((100, 100)),
                                    transforms.ToTensor()])

    positive = MergeImageDataset(anc_path, pos_path, neg_path, types=1, transform=transform)
    negtive = MergeImageDataset(anc_path, pos_path, neg_path, types=0, transform=transform)
    data = ConcatDataset([negtive, positive])
    num_samples = len(data)
    indices = list(range(num_samples))
    random.shuffle(indices)
    split = int(math.floor(0.3 * num_samples))  # Use 20% of data for validation
    train_indices, val_indices = indices[split:], indices[:split]

    # Create samplers for the training and validation parts
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(data, batch_size=32, sampler=train_sampler)
    val_loader = DataLoader(data, batch_size=32, sampler=val_sampler)

    return train_loader, val_loader


# package input of user
def get_test_data(test_input, test_pos):
    transform = transforms.Compose([transforms.Resize((100, 100)),
                                    transforms.ToTensor()])
    positive = TestDataset(test_input, test_pos, transform=transform)
    input_file_names = TestDataset(test_input, test_pos, transform=transform).GetFileName()

    test_loader_pos = DataLoader(positive)

    return test_loader_pos, input_file_names


def create_test_environment(root_path, pos_dir):
    INPUT_IMG = os.listdir(r"data\test\input")
    length = len(INPUT_IMG)

    POS_IMG = os.listdir(pos_dir)
    indices = list(range(len(POS_IMG)))
    random.shuffle(indices)
    pos_images = [POS_IMG[i] for i in indices[:length]]

    for pos_image in pos_images:
        source_path = os.path.join(pos_dir , pos_image)
        destination_path = os.path.join("data", "test", "positive")
        shutil.copy2(source_path, destination_path)




