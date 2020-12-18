import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, Bottleneck
from torchvision.models.utils import load_state_dict_from_url
#from torchsummary import summary

# in: [batch, 3, 224, 224]
# out: [batch, 2048, 7, 7]
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


# in: [batch, 2048, 7, 7]
# out: [batch, 2048]
def GD(x, p_k):
    x = x.view([-1, 2048, 49])
    x = torch.linalg.norm(x, ord=p_k, dim=2)
    return x

class GlobalDescriptor(nn.Module):
    def __init__(self, p_k):
        super().__init__()
        self.p_k = nn.Parameter(torch.FloatTensor(p_k))

    def forward(self, x):
        x = x.view([-1, 2048, 49])
        x = torch.linalg.norm(x, ord=self.p_k, dim=2)
        return x

# in: [batch, 2048]
# out: [batch, M]
# M: # of classes

# T: temperature
class AuxModule(nn.Module):
    def __init__(self, M, T):
        super().__init__()
        self.M = M
        self.T = T
        self.FC = nn.Linear(2048, M, bias=True)

    def forward(self, x):
        return torch.div(self.FC(x), self.T)

# in: [batch, 2048]
# out: [batch, k]
class RankingModule(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.FC = nn.Linear(2048, k, bias=False)

    def forward(self, x):
        x = self.FC(x)
        return torch.div(x, torch.linalg.norm(x)) # l2 norm
        


# k: ranking module output dimension
# n: # of GD
# M: # of class
# T: temperature
# p_k: GD parameter list

# in: [batch, 3, 224, 224]

class CGD(nn.Module):
    def __init__(self, k, n, M, T, p_k_list):
        super().__init__()
        self.k = k
        self.n = n
        self.M = M
        self.T = T
        self.p_k_list = p_k_list

        self.RankingLayers = nn.ModuleList([RankingModule(k) for _ in self.p_k_list])
        self.AuxLayer = AuxModule(M, T)
        self.ResnetBackbone = MyResNet(Bottleneck, [3, 4, 6, 3])

    def forward(self, x):

        x = self.ResnetBackbone(x)
        concatList = [self.RankingLayers[i](GD(x, p_k)) for i, p_k in enumerate(self.p_k_list)] # list of [p_k, batch, k]
        z = torch.cat(concatList, dim=1)

        return self.AuxLayer(GD(x, self.p_k_list[0])), torch.div(z, torch.linalg.norm(z)) # l2 norm

    
def testTensor(t):
    print(t)
    print(t.shape)



if __name__ == "__main__":
    
    model1 = CGD(1536, 1, 1024, 0.5, [1, 3, float('inf')])
    model2 = RankingModule(1536)


    testTensor(model1(torch.rand([10, 3, 224, 224])))

    #summary(model1, (3, 224, 224), device='cpu')


    #print(model1)



    






    









