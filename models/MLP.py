import torch
from torch import nn
from torch import Tensor
from IPDL import InformationPlane

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.layer1_IP = InformationPlane(beta=0.1, n_sigmas=200)
        self.layer2_IP = InformationPlane(beta=0.1, n_sigmas=200)
        self.layer3_IP = InformationPlane(beta=0.1, n_sigmas=200)
        self.layer4_IP = InformationPlane(beta=0.1, n_sigmas=200)
        self.layer5_IP = InformationPlane(beta=0.1, n_sigmas=200)

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