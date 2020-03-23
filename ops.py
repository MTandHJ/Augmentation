

import torch
import torchvision.transforms.functional as F
import PIL.ImageOps
import random
from torchvision import transforms
import scipy.stats
from collections.abc import Iterable


def special(cls, name, *args, **kwargs):
    class Newclass(cls):
        def __init__(self, *newargs, **newkwargs):
            newargs += args
            newkwargs.update(kwargs)
            super(Newclass, self).__init__(*newargs, **newkwargs)
    Newclass.__name__ = name
    return Newclass


class Hflip:
    """
    水平翻转
    """
    def __call__(self, img):
        return F.hflip(img)

class Vflip:
    """
    上下翻转
    """
    def __call__(self, img):
        return F.vflip(img)

class Equalize:
    """
    均衡图像直方图
    """
    def __init__(self, mask):
        self.mask = mask

    def __call__(self, img):
        return PIL.ImageOps.equalize(img, self.mask)

class Posterize:
    """
    减少每个颜色通道的位数
    """
    def __init__(self, bits):
        self.bits = bits

    def __call__(self, img):
        return PIL.ImageOps.posterize(img, self.bits)

class Grayscale:
    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        return F.to_grayscale(img, self.num_output_channels)

class Normalnoise:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def __call__(self, img):
        img = F.to_tensor(img)
        img += (torch.randn_like(img) + self.loc) * self.scale
        img = torch.clamp(img, 0., 1.)
        return F.to_pil_image(img)

class Uniformnoise:
    def __init__(self, alpha):
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha should be in [0, 1]...")
        self.alpha = alpha

    def __call__(self, img):
        img = F.to_tensor(img)
        img += torch.rand_like(img) * self.alpha
        img = torch.clamp(img, 0., 1.)
        return F.to_pil_image(img)

class Erase:

    def __init__(self, value):
        self.value = value

    def __call__(self, img):
        img = F.to_tensor(img)
        H, W = img.size()[-2:]
        lui = torch.randint(0, H, (1,)).item()
        luj = torch.randint(0, W, (1,)).item()
        rbi = torch.randint(lui, H, (1,)).item()
        rbj = torch.randint(luj, W, (1,)).item()
        h = rbj - luj
        w = rbi - lui

        return F.to_pil_image(F.erase(img, lui, luj, h, w, 0))

class Perspective:

    def __init__(self, startpoints, endpoints, interpolation=3):
        self.startpoints = startpoints
        self.endpoints = endpoints
        self.interpolation = interpolation

    def __call__(self, img):
        return F.perspective(img, self.startpoints,
                             self.endpoints, self.interpolation)


Translate = special(transforms.RandomAffine, "Translate", degrees = 0)
Scale = special(transforms.RandomAffine, "Scale", degrees = 0)
Shear = special(transforms.RandomAffine, "Shear", degrees=0)
Rotate = special(transforms.RandomAffine, "Rotate")
Brightness = special(transforms.ColorJitter, "Brightness")
Contrast = special(transforms.ColorJitter, "Contrast")
Saturation = special(transforms.ColorJitter, "Saturation")
Hue = special(transforms.ColorJitter, "Hue")
Crop = transforms.RandomCrop



class Augmix:

    def __init__(self, ops, k=3, alpha=1, beta=1):
        self.ops = ops
        self.k = k
        if isinstance(alpha, Iterable):
            self.alpha = alpha
        else:
            self.alpha = [alpha] * k
        self.beta = beta

    def get_params(self):
        op1, op2, op3 = random.sample(self.ops, 3)
        op12 = transforms.Compose([op1, op2])
        op123 = transforms.Compose([op1, op2, op3])
        return random.sample([op1, op12, op123], 1)[0]

    def __call__(self, img):
        weights = scipy.stats.dirichlet.rvs(self.alpha)[0]
        img_tensor = F.to_tensor(img)
        xaug = torch.zeros_like(img_tensor)
        for i in range(self.k):
            opschain = self.get_params()
            temp = weights[i] * F.to_tensor(opschain(img))
            xaug += temp
        m = scipy.stats.beta.rvs(self.beta, self.beta, size=1)[0]
        new_img = m * img_tensor + (1 - m) * xaug
        return F.to_pil_image(new_img)






