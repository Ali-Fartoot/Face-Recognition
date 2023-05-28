import os
import torchvision.transforms.functional as F
import torch
import random
import math
import uuid
from PIL import Image
import shutil
from pathlib import Path

root_path = Path(__file__).parent.parent



# Add paths for data
DATA_POOL_ADDRESS = os.path.join("data", "musk")

POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

# make directories
try:
    os.makedirs(POS_PATH)
except:
    print("you have already positive folder!")

try:
    os.makedirs(ANC_PATH)
except:
    print("you have already anchor (input) folder!")

print("Creating folders Done! [1/8]")


# Copy File
# os.system(cmd)


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
def indices_split_data(pool_address=DATA_POOL_ADDRESS):
    pool_images = os.listdir(pool_address)
    indices = list(range(len(pool_images)))
    random.shuffle(indices)
    split = int(math.floor(0.2 * len(pool_images)))
    anchor_indices, positive_indices = indices[split:], indices[:split]
    anchor_images = [pool_images[i] for i in anchor_indices]
    positive_images = [pool_images[i] for i in positive_indices]
    return anchor_images, positive_images


anchor_images, positive_images = indices_split_data()

for image_name in anchor_images:
    source_path = os.path.join(root_path,"app", DATA_POOL_ADDRESS, image_name)
    destination_path = os.path.join(root_path,"app", POS_PATH)
    shutil.copy2(source_path, destination_path)


for image_name in positive_images:
    source_path = os.path.join(root_path,"app", DATA_POOL_ADDRESS, image_name)
    destination_path = os.path.join(root_path,"app", ANC_PATH)
    shutil.copy2(source_path, destination_path)


print("Copying folders Done! [2/8]")



def processing_data(type):
    path = POS_PATH if type == "positive" else ANC_PATH
    count = 0
    files = os.listdir(os.path.join(path))
    length = 2500 - len(files)

    while count < length:
        for file_name in files:
            img_path = os.path.join(path, file_name)
            img = Image.open(img_path).convert('RGB')
            img_tensor = F.to_tensor(img)
            augmented_images = augmentation(img_tensor)
            for i, image in enumerate(augmented_images):
                image = F.to_pil_image(image)
                image.save(os.path.join(path, '{}.jpg'.format(uuid.uuid1())))
            else:
                count += 4

    files = os.listdir(os.path.join(path))
    while 2500 - len(files) > 0:
        file_name = files[len(files) - 4]
        img_path = os.path.join(path, file_name)
        img = Image.open(img_path).convert('RGB')
        img.save(os.path.join(path, '{}.jpg'.format(uuid.uuid1())))
        files = os.listdir(os.path.join(path))


processing_data("positive")
processing_data("anchor")

print("Augmentation datas Done! [3/8]")