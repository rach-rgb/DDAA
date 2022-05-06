# augment tensor type images directly
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF

random_mirror = True  # flip magnitude by 1/2 probability


def ShearX(img, v):     # [-0.2, 0.2]
    assert -0.2 <= v <= 0.2
    v = v * 180
    return TF.affine(img, 0, [0, 0], 1.0, [v, 0.0])


def ShearY(img, v):     # [-0.2, 0.2]
    assert -0.2 <= v <= 0.2
    v = v * 180
    return TF.affine(img, 0, [0, 0], 1.0, [0.0, v])


def TranslateX(img, v):     # (-0.45, 0.45)
    assert -0.45 <= v <= 0.45
    v = v * img.shape[1]
    return TF.affine(img, 0, [v, 0], 1.0, [0.0, 0.0])


def TranslateY(img, v):     # (-0.45, 0.45)
    assert -0.45 <= v <= 0.45
    v = v * img.shape[1]
    return TF.affine(img, 0, [0, v], 1.0, [0.0, 0.0])


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    return TF.rotate(img, v)


def AutoContrast(img, v):  # [0, 1]
    assert 0.0 <= v <= 1.0
    return TF.autocontrast(img)


def Invert(img, _):
    return TF.invert(img)


def Equalize(img, _):
    return TF.equalize((img * 255).type(torch.uint8))


def Flip(img, _):
    return transforms.RandomHorizontalFlip(p=1)(img)


def Solarize(img, v):   # [0, 255]
    assert 0 <= v <= 255
    return TF.solarize((img * 255).type(torch.uint8), v)


def Posterize(img, v):  # [4, 8]
    assert 4 <= v <= 8
    v = int(v)
    return TF.posterize((img * 255).type(torch.uint8), v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return TF.adjust_contrast(img, v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return TF.adjust_saturation(img, v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return TF.adjust_brightness(img, v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return TF.adjust_sharpness(img, v)


def Identity(img, _):
    return img




AUGMENT_LIST = [
        (ShearX, -0.2, 0.2),  # 0
        (ShearY, -0.2, 0.2),  # 1
        (TranslateX, -0.45, 0.45),  # 2
        (TranslateY, -0.45, 0.45),  # 3
        (Rotate, -30, 30),  # 4
        (AutoContrast, 0, 1),  # 5
        (Invert, 0, 1),  # 6
        (Equalize, 0, 1),  # 7
        (Solarize, 0, 255),  # 8
        (Posterize, 4, 8),  # 9
        (Contrast, 0.1, 1.9),  # 10
        (Color, 0.1, 1.9),  # 11
        (Brightness, 0.1, 1.9),  # 12
        (Sharpness, 0.1, 1.9),  # 13
        (Flip, 0, 1),  # 15
        (Identity, 0, 1),   # 16
        ]


# return dictionary of augmentation operation
def get_augment(name):
    augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in AUGMENT_LIST}
    return augment_dict[name]


# parameters: img(Tensor image), name(operation name), level(magnitude 0~1)
def apply_augment(img, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(img, level * (high - low) + low)