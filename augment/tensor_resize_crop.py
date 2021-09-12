# Adds support for resizing and cropping `TensorImage` and `TensorMask` as an `item_tfm`

from torchvision.transforms.functional import pad as tvpad
from torchvision.transforms.functional import _interpolation_modes_from_int
from torch.nn.functional import interpolate
from fastai.vision.augment import RandomCrop, CropPad, Resize, PadMode, RandomResizedCrop, ResizeMethod, _pad_modes, _get_sz
from fastai.vision.core import TensorImage, TensorMask
from fastcore.basics import fastuple
from fastcore.foundation import patch
from PIL import Image
import operator

def _resize(x, size, shape, interpolation):
    align_corners = False if interpolation in ["bilinear", "bicubic"] else None
    return interpolate(x.view(shape), size=size, mode=interpolation, align_corners=align_corners)

@patch
def resize(x: (TensorImage, TensorMask), size, interpolation):
    if len(x.shape)==3: l, c, h, w = 3, x.shape[0], x.shape[1], x.shape[2]
    elif len(x.shape)==2: l, c, h, w = 2, 1, x.shape[0], x.shape[1]

    if len(size)==2: sh, sw = size[0], size[1]
    elif len(size)==1: sh, sw = size, size

    x = _resize(x, size=size, shape=[1,c,h,w], interpolation=interpolation.value)

    if l==3: return x.view(c, sh, sw)
    else: return x.view(sh, sw)

@patch
def _do_crop_pad(x: (TensorImage, TensorMask), sz, tl, orig_sz,
                 pad_mode=PadMode.Zeros, resize_mode=Image.BILINEAR, resize_to=None):
    # PyTorch and PIL axis are opposite, need to reverse PIL axis input for crop and resize
    if any(tl.ge(0)) or any(tl.add(sz).le(orig_sz)):
        # At least one dim is inside the image, so needs to be cropped
        c = tl.max(0)
        left, top, right, bottom = *c, *tl.add(sz).min(orig_sz)
        x = x[..., top:bottom, left:right]
    if any(tl.lt(0)) or any(tl.add(sz).ge(orig_sz)):
        # At least one dim is outside the image, so needs to be padded
        p = (-tl).max(0)
        f = (sz-orig_sz).add(tl).max(0)
        if len(x.shape)==2: 
            x = x.view(1, x.shape[0], x.shape[1])
            x = tvpad(x, (*p, *f), padding_mode=_pad_modes[pad_mode])
            x = x.view(x.shape[1], x.shape[2])
        else:
            x = tvpad(x, (*p, *f), padding_mode=_pad_modes[pad_mode])
    if resize_to is not None:
        resize_mode = Image.NEAREST if isinstance(x,TensorMask) else resize_mode
        x = x.resize([*resize_to][::-1], _interpolation_modes_from_int(resize_mode))
    return x

@patch
def crop_pad(x: (TensorImage, TensorMask),
             sz, tl=None, orig_sz=None, pad_mode=PadMode.Zeros, resize_mode=Image.BILINEAR, resize_to=None):
    if isinstance(sz,int): sz = (sz,sz)
    orig_sz = fastuple(_get_sz(x) if orig_sz is None else orig_sz)
    sz,tl = fastuple(sz),fastuple(((_get_sz(x)-sz)//2) if tl is None else tl)
    return x._do_crop_pad(sz, tl, orig_sz=orig_sz, pad_mode=pad_mode, resize_mode=resize_mode, resize_to=resize_to)

@patch
def encodes(self:RandomCrop, x: (TensorImage, TensorMask)):
    return x.crop_pad(self.size, self.tl, orig_sz=self.orig_sz)

@patch
def encodes(self:CropPad, x: (TensorImage, TensorMask)):
    orig_sz = _get_sz(x)
    tl = (orig_sz-self.size)//2
    return x.crop_pad(self.size, tl, orig_sz=orig_sz, pad_mode=self.pad_mode)

@patch
def encodes(self:Resize, x: (TensorImage, TensorMask)):
    orig_sz = _get_sz(x)
    if self.method==ResizeMethod.Squish:
        return x.crop_pad(orig_sz, fastuple(0,0), orig_sz=orig_sz, pad_mode=self.pad_mode,
                resize_mode=self.mode_mask if isinstance(x,TensorMask) else self.mode, resize_to=self.size)

    w,h = orig_sz
    op = (operator.lt,operator.gt)[self.method==ResizeMethod.Pad]
    m = w/self.size[0] if op(w/self.size[0],h/self.size[1]) else h/self.size[1]
    cp_sz = (int(m*self.size[0]),int(m*self.size[1]))
    tl = fastuple(int(self.pcts[0]*(w-cp_sz[0])), int(self.pcts[1]*(h-cp_sz[1])))
    return x.crop_pad(cp_sz, tl, orig_sz=orig_sz, pad_mode=self.pad_mode,
                resize_mode=self.mode_mask if isinstance(x,TensorMask) else self.mode, resize_to=self.size)


@patch
def encodes(self:RandomResizedCrop, x: (TensorImage, TensorMask)):
    res = x.crop_pad(self.cp_size, self.tl, orig_sz=self.orig_sz,
        resize_mode=self.mode_mask if isinstance(x,TensorMask) else self.mode, resize_to=self.final_size)
    if self.final_size != self.size: res = res.crop_pad(self.size) #Validation set: one final center crop
    return res
