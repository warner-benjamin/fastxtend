# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/vision.models.resnet_blocks.ipynb (unless otherwise specified).

__all__ = ['ResBlock', 'ResNeXtBlock', 'SEBlock', 'SEResNeXtBlock', 'ECABlock', 'ECAResNeXtBlock', 'SABlock',
           'SAResNeXtBlock', 'TABlock', 'TAResNeXtBlock']

# Cell
import math
import torch.nn as nn
from functools import partial
from fastcore.meta import delegates
from fastai.torch_core import Module
from fastai.basics import defaults
from fastai.layers import ConvLayer, NormType, SimpleSelfAttention, AvgPool, SEModule
from torchvision.ops.stochastic_depth import StochasticDepth
from .attention_modules import *

# Cell
class ResBlock(Module):
    "Resnet block from `ni` to `nh` with `stride`"
    @delegates(ConvLayer.__init__)
    def __init__(self, expansion, ni, nf, stride=1, groups=1, attn_mod=None, nh1=None, nh2=None,
                 dw=False, g2=1, sa=False, sym=False, norm_type=NormType.Batch, act_cls=defaults.activation,
                 ndim=2, ks=3, block_pool=AvgPool, pool_first=True, stoch_depth=0, **kwargs):
        norm2 = (NormType.BatchZero if norm_type==NormType.Batch else
                 NormType.InstanceZero if norm_type==NormType.Instance else norm_type)
        if nh2 is None: nh2 = nf
        if nh1 is None: nh1 = nh2
        nf,ni = nf*expansion,ni*expansion
        k0 = dict(norm_type=norm_type, act_cls=act_cls, ndim=ndim, **kwargs)
        k1 = dict(norm_type=norm2, act_cls=None, ndim=ndim, **kwargs)
        convpath  = [ConvLayer(ni,  nh2, ks, stride=stride, groups=ni if dw else groups, **k0),
                     ConvLayer(nh2,  nf, ks, groups=g2, **k1)
        ] if expansion == 1 else [
                     ConvLayer(ni,  nh1, 1, **k0),
                     ConvLayer(nh1, nh2, ks, stride=stride, groups=nh1 if dw else groups, **k0),
                     ConvLayer(nh2,  nf, 1, groups=g2, **k1)]
        if attn_mod: convpath.append(attn_mod(nf))
        if sa: convpath.append(SimpleSelfAttention(nf,ks=1,sym=sym))
        self.convpath = nn.Sequential(*convpath)
        idpath = []
        if ni!=nf: idpath.append(ConvLayer(ni, nf, 1, act_cls=None, ndim=ndim, **kwargs))
        if stride!=1:
            idpath.insert((1,0)[pool_first], block_pool(stride, ndim=ndim, ceil_mode=True))
        self.idpath = nn.Sequential(*idpath)
        self.act = act_cls(inplace=True) if act_cls is defaults.activation else act_cls()
        self.depth = nn.Identity() if stoch_depth==0 else StochasticDepth(stoch_depth, 'batch')

    def forward(self, x): return self.act(self.depth(self.convpath(x)) + self.idpath(x))

# Cell
def ResNeXtBlock(expansion, ni, nf, groups=32, stride=1, base_width=4, **kwargs):
    w = math.floor(nf * (base_width / 64)) * groups
    return ResBlock(expansion, ni, nf, stride=stride, groups=groups, nh2=w, **kwargs)

# Cell
def SEBlock(expansion, ni, nf, groups=1, se_reduction=16, stride=1, se_act_cls=defaults.activation, **kwargs):
    attn_mod = partial(SEModule, reduction=se_reduction, act_cls=se_act_cls)
    return ResBlock(expansion, ni, nf, stride=stride, groups=groups, attn_mod=attn_mod, nh1=nf*2, nh2=nf*expansion, **kwargs)

# Cell
def SEResNeXtBlock(expansion, ni, nf, groups=32, se_reduction=16, stride=1, base_width=4, se_act_cls=defaults.activation, **kwargs):
    w = math.floor(nf * (base_width / 64)) * groups
    attn_mod = partial(SEModule, reduction=se_reduction, act_cls=se_act_cls)
    return ResBlock(expansion, ni, nf, stride=stride, groups=groups, attn_mod=attn_mod, nh2=w, **kwargs)

# Cell
def ECABlock(expansion, ni, nf, groups=1, eca_ks=None, stride=1, **kwargs):
    attn_mod = partial(ECA, ks=eca_ks)
    return ResBlock(expansion, ni, nf, stride=stride, groups=groups, attn_mod=attn_mod, nh1=nf*2, nh2=nf*expansion, **kwargs)

# Cell
def ECAResNeXtBlock(expansion, ni, nf, groups=32, eca_ks=None, stride=1, base_width=4, **kwargs):
    w = math.floor(nf * (base_width / 64)) * groups
    attn_mod = partial(ECA, ks=eca_ks)
    return ResBlock(expansion, ni, nf, stride=stride, groups=groups, attn_mod=attn_mod, nh2=w, **kwargs)

# Cell
def SABlock(expansion, ni, nf, groups=1, sa_grps=64, stride=1, **kwargs):
    attn_mod = partial(ShuffleAttention, groups=sa_grps)
    return ResBlock(expansion, ni, nf, stride=stride, groups=groups, attn_mod=attn_mod, nh1=nf*2, nh2=nf*expansion, **kwargs)

# Cell
def SAResNeXtBlock(expansion, ni, nf, groups=32, sa_grps=64, stride=1, base_width=4, **kwargs):
    w = math.floor(nf * (base_width / 64)) * groups
    attn_mod = partial(ShuffleAttention, groups=sa_grps)
    return ResBlock(expansion, ni, nf, stride=stride, groups=groups, attn_mod=attn_mod, nh2=w, **kwargs)

# Cell
def TABlock(expansion, ni, nf, groups=1, ta_ks=7, stride=1, **kwargs):
    attn_mod = partial(TripletAttention, ks=ta_ks)
    return ResBlock(expansion, ni, nf, stride=stride, groups=groups, attn_mod=attn_mod, nh1=nf*2, nh2=nf*expansion, **kwargs)

# Cell
def TAResNeXtBlock(expansion, ni, nf, groups=32, ta_ks=7, stride=1, base_width=4, **kwargs):
    w = math.floor(nf * (base_width / 64)) * groups
    attn_mod = partial(TripletAttention, ks=ta_ks)
    return ResBlock(expansion, ni, nf, stride=stride, groups=groups, attn_mod=attn_mod, nh2=w, **kwargs)