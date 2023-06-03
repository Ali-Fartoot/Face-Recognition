import torch
from model import SiameseNetwork
import math


def round(data):
    for i, value in enumerate(data):
        data[i] = 0 if value < 0.5 else 1
    return data


def testing(input_loader, file_name, name):
    with torch.no_grad():
        torch.manual_seed(34)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        siamese_net = torch.jit.load('model.pt').to(device)

        for i, (input_image, person_image) in enumerate(input_loader):
            input_image = input_image.to(device)
            person_image = person_image.to(device)
            output = siamese_net(input_image, person_image)
            accuracy = ((math.fabs(0.5 - output.item()) / 0.5) * 100)
            result = round(output)
            print(f"{file_name[i]} is {name}" if result == 1 else f"{file_name[i]} is not {name}", f" Accuracy:{accuracy: .2f} % ", flush=True)
