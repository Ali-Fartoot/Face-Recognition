from torch import nn
import torch


# model
class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()

        self.conv_1 = nn.Conv2d(3, 64, kernel_size=(10, 10))
        self.relu_1 = nn.ReLU()
        self.maxpool_1 = nn.MaxPool2d(2, stride=2, padding=1)

        self.conv_2 = nn.Conv2d(64, 128, kernel_size=(7, 7))
        self.relu_2 = nn.ReLU()
        self.maxpool_2 = nn.MaxPool2d(2, 2)

        self.conv_3 = nn.Conv2d(128, 128, kernel_size=(4, 4))
        self.relu_3 = nn.ReLU()
        self.maxpool_3 = nn.MaxPool2d(2, 2, padding=1)

        self.conv_4 = nn.Conv2d(128, 256, kernel_size=(4, 4))
        self.relu_4 = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=1)

        self.normalization_64 = nn.BatchNorm2d(64)
        self.normalization_128 = nn.BatchNorm2d(128)

        self.linear = nn.Linear(256 * 6 * 6, 4096)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.maxpool_1(x)
        x = self.normalization_64(x)

        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.maxpool_2(x)
        x = self.normalization_128(x)

        x = self.conv_3(x)
        x = self.relu_3(x)
        x = self.maxpool_3(x)
        x = self.normalization_128(x)

        x = self.conv_4(x)
        x = self.relu_4(x)

        x = self.flatten(x)
        x = self.linear(x)

        return x


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.embeder = Embedding()
        self.classifier = nn.Linear(4096, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_image, validation_image):
        input_image = self.embeder(input_image)
        validation_image = self.embeder(validation_image)
        result = torch.abs(input_image - validation_image)
        x = self.classifier(result)
        x = self.sigmoid(x)
        return x
