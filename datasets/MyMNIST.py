import os 
from PIL import Image, ImageOps
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, ToPILImage, Normalize
from torch.nn.functional import one_hot
from torch.distributions.dirichlet import Dirichlet


def one_hot_dirichlet(x, num_classes=-1):
    labels = one_hot(x, num_classes=num_classes).float()
    labels = (labels * 10000) + 100
    distribution = Dirichlet(labels)
    return distribution.sample()

class MyMNIST():
    '''
    @param dataset_path: Path where is located the dataset.
    '''
    def __init__(self, train_set_split = 0.75):
        self.transformToTensor = Compose([ ToTensor(), Normalize((0.1307,), (0.3081,))])
        self.transformToImage = Compose([ Normalize((-0.1307/0.3081,), (1/0.3081,)), ToPILImage()])
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dataset = MNIST(os.path.join(dir_path, "MNIST"), train=True, download=True, transform=self.transformToTensor)
        
        data = dataset.data
        labels = dataset.targets
        
        train_len = (int)(train_set_split * len(data))
    
        self.training = True
        self.train_set = (data[:train_len], one_hot(labels[:train_len]))
        self.eval_set = (data[train_len:], one_hot_dirichlet(labels[train_len:]))
    

    def __len__(self):
        if self.training:
            return len(self.train_set[0])
        else:
            return len(self.eval_set[0])
        
    def __getitem__(self, idx):
        if self.training:
            img, target = self.train_set[0][idx], self.train_set[1][idx]
        else:
            img, target = self.eval_set[0][idx], self.eval_set[1][idx]

        img = Image.fromarray(img.numpy(), mode='L')
        img = self.transformToTensor(img)

        return img, target


    def eval(self):
        self.training = False

    def train(self):
        self.training = True
