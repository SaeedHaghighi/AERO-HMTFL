



import torch
import torch.nn as nn
import torch.nn.functional as F


class LightCNN(nn.Module):






    def __init__(self, num_classes: int = 10) -> None:






        super().__init__()


        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)


        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)


        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)


        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:









        x = self.pool(F.relu(self.bn1(self.conv1(x))))


        x = self.pool(F.relu(self.bn2(self.conv2(x))))


        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)

        return x
