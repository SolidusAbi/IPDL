import torch
from torch import nn
from torch import Tensor


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1_IP = InformationPlane(beta=0.1, n_sigmas=150)
        self.layer2_IP = InformationPlane(beta=0.1, n_sigmas=150)
        self.layer3_IP = InformationPlane(beta=0.1, n_sigmas=150 )
        self.layer4_IP = InformationPlane(beta=0.1, n_sigmas=150 )
        self.fc_IP = InformationPlane(beta=0.1, n_sigmas=150)


        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 4, 3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            self.layer1_IP,
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 8, 3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            self.layer2_IP,
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(8, 16, 3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            self.layer3_IP,
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(400, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            self.layer4_IP,
            nn.Linear(256, 10)
        )

        
        self.softmax = torch.nn.Softmax()

    def forward(self, x: Tensor, labels=None) -> Tensor:
        if not(self.training):
            [ip.setInputLabel(x, labels) for ip in self.getInformationPlaneLayers()]
        

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        self.fc_IP(self.softmax(x))
        return x

    def getInformationPlaneLayers(self) -> list:
        return [self.layer1_IP, self.layer2_IP, self.layer3_IP, self.layer4_IP, self.fc_IP]

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.layer1_IP = InformationPlane(beta=0.1, n_sigmas=150)
        self.layer2_IP = InformationPlane(beta=0.1, n_sigmas=150)
        self.layer3_IP = InformationPlane(beta=0.1, n_sigmas=150)
        self.layer4_IP = InformationPlane(beta=0.1, n_sigmas=150)
        self.layer5_IP = InformationPlane(beta=0.1, n_sigmas=150)

        self.layer1 = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            self.layer1_IP,
        )

        self.layer2 = nn.Sequential(
            nn.Linear(1024, 20),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(20),
            self.layer2_IP
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(20, 20),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(20),
            self.layer3_IP
        )

        self.layer4 = nn.Sequential(
            nn.Linear(20, 20),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(20),
            self.layer4_IP
        )

        self.layer5 = nn.Sequential(
            nn.Linear(20, 10)
        )


        self.softmax = torch.nn.Softmax()

    def forward(self, x: Tensor, labels=None) -> Tensor:
        if not(self.training):
            [ip.setInputLabel(x, labels) for ip in self.getInformationPlaneLayers()]

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        self.layer5_IP(self.softmax(x))

        return x

    def getInformationPlaneLayers(self) -> list:
        return [self.layer1_IP, self.layer2_IP, self.layer3_IP, self.layer4_IP, self.layer5_IP]