import torch
import torchvision

vgg16pre=torchvision.models.vgg16(pretrained=False)
vgg16pre.load_state_dict(torch.load("vgg16_pretrain.pth"), False)
# vgg16pre.load_state_dict(torch.load("vgg16_pretrain.pth"))
print(vgg16pre)