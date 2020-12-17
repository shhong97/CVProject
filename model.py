import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, Bottleneck
from torchvision.models.utils import load_state_dict_from_url
from pytorch_metric_learning import losses
from torchsummary import summary

# in: [1, 3, 224, 224]
# out: [1, 2048, 7, 7]
class MyResNet(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth',)
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        self.load_state_dict(state_dict)

    # remove FC layer (https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L243-L244)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# in: [1, 2048, 7, 7]
# out: [2048]
def GD(x, p_k):
    x = torch.pow(x, exponent=p_k) # element-wise power [1, 2048, 7, 7]
    x = torch.mean(x, dim=[2, 3]) # mean 7x7 [1, 2048]
    x = torch.pow(x, exponent=1.0/p_k) # p_k root square [1, 2048]
    return x[0]

class GlobalDescriptor(nn.Module):
    def __init__(self, p_k):
        super().__init__()
        self.p_k = p_k

    def forward(self, x):
        x = torch.pow(x, exponent=self.p_k) # element-wise power [1, 2048, 7, 7]
        x = torch.mean(x, dim=[2, 3]) # mean 7x7 [1, 2048]
        x = torch.pow(x, exponent=1.0/self.p_k) # p_k root square [1, 2048]
        return x[0]

# in: [2048]
# out: [M]
class AuxModule(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
        self.BN = nn.BatchNorm1d(2048)
        self.FC = nn.Linear(2048, M)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.BN(x)
        x = self.FC(x)
        x = self.softmax(x)
        return x

# in: [2048]
# out: [k]
class RankingModule(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.FC = nn.Linear(2048, k, bias=False)

    def forward(self, x):
        x = self.FC(x)
        return torch.div(x, torch.linalg.norm(x)) # l2 norm
        


# k: ranking module output dimension
# n: # of GD
# M: aux module output dimension
# p_k: GD parameter list

# in: [1, 3, 224, 224]

class CGD(nn.Module):
    def __init__(self, k, n, M, p_k_list):
        super().__init__()
        self.k = k
        self.n = n
        self.M = M
        self.p_k_list = p_k_list

        self.RankingLayers = nn.ModuleList([RankingModule(k) for _ in self.p_k_list])
        #self.AuxLayer = AuxModule(M)
        self.ResnetBackbone = MyResNet(Bottleneck, [3, 4, 6, 3])

    def forward(self, x):
        concatList = []
        x = self.ResnetBackbone(x)
        for i, p_k in enumerate(self.p_k_list):
            concatList.append(self.RankingLayers[i](GD(x, p_k)))

        z = torch.cat(concatList)
        return torch.div(z, torch.linalg.norm(z)) # l2 norm

    
def testTensor(t):
    print(t)
    print(t.shape)





if __name__ == "__main__":
    
    model1 = CGD(1536, 1, 1024, [1, 2, 3])
    summary(model1, (3, 224, 224), device='cpu')


    #print(model1)



    






    









