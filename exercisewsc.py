import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

img_path = "./data/voc12/JPEGImages/2007_000027.jpg"
img = Image.open(img_path)
print(img)


trans = transforms.Compose([transforms.ToTensor()])
img=trans(img)
img=torch.reshape(img,(1,3,500,486))

class Shichao(nn.Module):
    def __init__(self):
        super(Shichao,self).__init__()
        self.conv=Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1,
                          padding=0)
        self.poo=MaxPool2d(kernel_size=5, stride=1, padding=0)
    def forward(self, x):
        x=self.conv(x)
        x=self.poo(x)
        return x

shichao=Shichao()

output = shichao(img)
print(img.shape)
print(output.shape)

writer=SummaryWriter("./logs")

writer.add_images("input", img)
writer.add_images("output", output)

writer.close()
