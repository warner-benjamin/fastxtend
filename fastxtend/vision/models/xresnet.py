# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/vision.models.xresnet.ipynb (unless otherwise specified).

__all__ = ['ResBlock', 'ResNeXtBlock', 'SEBlock', 'SEResNeXtBlock', 'ECABlock', 'ECAResNeXtBlock', 'SABlock',
           'SAResNeXtBlock', 'TABlock', 'TAResNeXtBlock', 'XResNet', 'xresnet18', 'xresnet34', 'xresnet50',
           'xresnet101', 'xresnext18', 'xresnext34', 'xresnext50', 'xresnext101', 'xse_resnet18', 'xse_resnet34',
           'xse_resnet50', 'xse_resnet101', 'xse_resnext18', 'xse_resnext34', 'xse_resnext50', 'xse_resnext101',
           'xeca_resnet18', 'xeca_resnet34', 'xeca_resnet50', 'xeca_resnet101', 'xeca_resnext18', 'xeca_resnext34',
           'xeca_resnext50', 'xeca_resnext101', 'xsa_resnet18', 'xsa_resnet34', 'xsa_resnet50', 'xsa_resnet101',
           'xsa_resnext18', 'xsa_resnext34', 'xsa_resnext50', 'xsa_resnext101', 'xta_resnet18', 'xta_resnet34',
           'xta_resnet50', 'xta_resnet101', 'xta_resnext18', 'xta_resnext34', 'xta_resnext50', 'xta_resnext101']

# Cell
from torchvision.ops.stochastic_depth import StochasticDepth

from fastai.basics import defaults
from fastai.layers import ConvLayer, NormType, SimpleSelfAttention, AvgPool, SEModule, MaxPool, AdaptiveAvgPool, Flatten
from fastai.vision.models.xresnet import init_cnn

from .attention_modules import *
from ...imports import *

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
@delegates(ResBlock)
def ResNeXtBlock(expansion, ni, nf, groups=32, stride=1, base_width=4, **kwargs):
    w = math.floor(nf * (base_width / 64)) * groups
    return ResBlock(expansion, ni, nf, stride=stride, groups=groups, nh2=w, **kwargs)

# Cell
@delegates(ResBlock)
def SEBlock(expansion, ni, nf, groups=1, se_reduction=16, stride=1, se_act_cls=defaults.activation, **kwargs):
    "A Squeeze and Excitation `XResNet` Block. Can set `se_act_cls` seperately."
    attn_mod = partial(SEModule, reduction=se_reduction, act_cls=se_act_cls)
    return ResBlock(expansion, ni, nf, stride=stride, groups=groups, attn_mod=attn_mod, nh1=nf*2, nh2=nf*expansion, **kwargs)

# Cell
@delegates(ResBlock)
def SEResNeXtBlock(expansion, ni, nf, groups=32, se_reduction=16, stride=1, base_width=4, se_act_cls=defaults.activation, **kwargs):
    "A Squeeze and Excitation `XResNeXtBlock`. Can set `se_act_cls` seperately."
    w = math.floor(nf * (base_width / 64)) * groups
    attn_mod = partial(SEModule, reduction=se_reduction, act_cls=se_act_cls)
    return ResBlock(expansion, ni, nf, stride=stride, groups=groups, attn_mod=attn_mod, nh2=w, **kwargs)

# Cell
@delegates(ResBlock)
def ECABlock(expansion, ni, nf, groups=1, eca_ks=None, stride=1, **kwargs):
    "An Efficient Channel Attention `XResNet` Block"
    attn_mod = partial(ECA, ks=eca_ks)
    return ResBlock(expansion, ni, nf, stride=stride, groups=groups, attn_mod=attn_mod, nh1=nf*2, nh2=nf*expansion, **kwargs)

# Cell
@delegates(ResBlock)
def ECAResNeXtBlock(expansion, ni, nf, groups=32, eca_ks=None, stride=1, base_width=4, **kwargs):
    "An Efficient Channel Attention `XResNeXtBlock`"
    w = math.floor(nf * (base_width / 64)) * groups
    attn_mod = partial(ECA, ks=eca_ks)
    return ResBlock(expansion, ni, nf, stride=stride, groups=groups, attn_mod=attn_mod, nh2=w, **kwargs)

# Cell
@delegates(ResBlock)
def SABlock(expansion, ni, nf, groups=1, sa_grps=64, stride=1, **kwargs):
    "A Shuffle Attention `XResNet` Block"
    attn_mod = partial(ShuffleAttention, groups=sa_grps)
    return ResBlock(expansion, ni, nf, stride=stride, groups=groups, attn_mod=attn_mod, nh1=nf*2, nh2=nf*expansion, **kwargs)

# Cell
@delegates(ResBlock)
def SAResNeXtBlock(expansion, ni, nf, groups=32, sa_grps=64, stride=1, base_width=4, **kwargs):
    "A Shuffle Attention `XResNeXtBlock`"
    w = math.floor(nf * (base_width / 64)) * groups
    attn_mod = partial(ShuffleAttention, groups=sa_grps)
    return ResBlock(expansion, ni, nf, stride=stride, groups=groups, attn_mod=attn_mod, nh2=w, **kwargs)

# Cell
@delegates(ResBlock)
def TABlock(expansion, ni, nf, groups=1, ta_ks=7, stride=1, **kwargs):
    "A Triplet Attention `XResNet` Block"
    attn_mod = partial(TripletAttention, ks=ta_ks)
    return ResBlock(expansion, ni, nf, stride=stride, groups=groups, attn_mod=attn_mod, nh1=nf*2, nh2=nf*expansion, **kwargs)

# Cell
@delegates(ResBlock)
def TAResNeXtBlock(expansion, ni, nf, groups=32, ta_ks=7, stride=1, base_width=4, **kwargs):
    "A Triplet Attention `XResNeXtBlock`"
    w = math.floor(nf * (base_width / 64)) * groups
    attn_mod = partial(TripletAttention, ks=ta_ks)
    return ResBlock(expansion, ni, nf, stride=stride, groups=groups, attn_mod=attn_mod, nh2=w, **kwargs)

# Cell
class XResNet(nn.Sequential):
    "A flexible version of fastai's XResNet"
    @delegates(ResBlock)
    def __init__(self, block, expansion, layers, p=0.0, c_in=3, n_out=1000, stem_szs=(32,32,64),
                 block_szs=[64,128,256,512], widen=1.0, sa=False, act_cls=defaults.activation, ndim=2,
                 ks=3, stride=2, stem_layer=ConvLayer, stem_pool=MaxPool, head_pool=AdaptiveAvgPool,
                 custom_head=None, **kwargs):
        store_attr('block,expansion,act_cls,ndim,ks')
        if ks % 2 == 0: raise Exception('kernel size has to be odd!')
        stem_szs = [c_in, *stem_szs]
        stem = [stem_layer(stem_szs[i], stem_szs[i+1], ks=ks, stride=stride if i==0 else 1,
                           act_cls=act_cls, ndim=ndim)
                for i in range(3)]

        assert len(layers) == len(block_szs), 'Length of `layers` must match `block_szs`'
        block_szs = [int(o*widen) for o in block_szs]
        block_szs = [stem_szs[-1]//expansion] + block_szs
        stem_pool = stem_pool(ks=ks, stride=stride, padding=ks//2, ndim=ndim)
        if not is_listy(stem_pool): stem_pool = [stem_pool]
        blocks    = self._make_blocks(layers, block_szs, sa, stride, **kwargs)

        if custom_head:
            head = custom_head(block_szs[-1]*expansion, n_out)
            if not is_listy(head): head = [head]
            body = nn.Sequential(*stem, *stem_pool, *blocks)
            init_cnn(body)
            super().__init__(*list(body), *head)
        else:
            head = self._make_head(block_szs[-1]*expansion, head_pool, ndim, p, n_out)
            super().__init__(*stem, *stem_pool, *blocks, *head)
            init_cnn(self)

    def _make_blocks(self, layers, block_szs, sa, stride, **kwargs):
        return [self._make_layer(ni=block_szs[i], nf=block_szs[i+1], blocks=l,
                                 stride=1 if i==0 else stride, sa=sa and i==len(layers)-4, **kwargs)
                for i,l in enumerate(layers)]

    def _make_layer(self, ni, nf, blocks, stride, sa, **kwargs):
        return nn.Sequential(
            *[self.block(self.expansion, ni if i==0 else nf, nf, stride=stride if i==0 else 1,
                      sa=sa and i==(blocks-1), act_cls=self.act_cls, ndim=self.ndim, ks=self.ks, **kwargs)
              for i in range(blocks)])

    def _make_head(self, ni, head_pool, ndim, p, n_out):
        return [head_pool(sz=1, ndim=ndim), nn.Flatten(), nn.Dropout(p), nn.Linear(ni, n_out)]

# Cell
@delegates(XResNet)
def xresnet18(n_out=1000, c_in=3, p=0.0, act_cls=defaults.activation, **kwargs):
    return XResNet(ResBlock, 1, [2, 2, 2, 2], n_out=n_out, **kwargs)

@delegates(XResNet)
def xresnet34(n_out=1000, **kwargs):
    return XResNet(ResBlock, 1, [3, 4, 6, 3], n_out=n_out, **kwargs)

@delegates(XResNet)
def xresnet50(n_out=1000, **kwargs):
    return XResNet(ResBlock, 4, [3, 4, 6, 3], n_out=n_out, **kwargs)

@delegates(XResNet)
def xresnet101(n_out=1000, **kwargs):
    return XResNet(ResBlock, 4, [3, 4, 23, 3], n_out=n_out, **kwargs)

# Cell
@delegates(XResNet)
def xresnext18(n_out=1000, **kwargs):
    return XResNet(ResNeXtBlock, 1, [2, 2, 2, 2], n_out=n_out, **kwargs)

@delegates(XResNet)
def xresnext34(n_out=1000, **kwargs):
    return XResNet(ResNeXtBlock, 1, [3, 4, 6, 3], n_out=n_out, **kwargs)

@delegates(XResNet)
def xresnext50(n_out=1000, **kwargs):
    return XResNet(ResNeXtBlock, 4, [3, 4, 6, 3], n_out=n_out, **kwargs)

@delegates(XResNet)
def xresnext101(n_out=1000, **kwargs):
    return XResNet(ResNeXtBlock, 4, [3, 4, 23, 3], n_out=n_out, **kwargs)

# Cell
@delegates(XResNet)
def xse_resnet18(n_out=1000, **kwargs):
    return XResNet(SEBlock, 1, [2, 2, 2, 2], n_out=n_out, **kwargs)

@delegates(XResNet)
def xse_resnet34(n_out=1000, **kwargs):
    return XResNet(SEBlock, 1, [3, 4, 6, 3], n_out=n_out, **kwargs)

@delegates(XResNet)
def xse_resnet50(n_out=1000, **kwargs):
    return XResNet(SEBlock, 4, [3, 4, 6, 3], n_out=n_out, **kwargs)

@delegates(XResNet)
def xse_resnet101(n_out=1000, **kwargs):
    return XResNet(SEBlock, 4, [3, 4, 23, 3], n_out=n_out, **kwargs)

# Cell
@delegates(XResNet)
def xse_resnext18(n_out=1000, **kwargs):
    return XResNet(SEResNeXtBlock, 1, [2, 2, 2, 2], n_out=n_out, **kwargs)

@delegates(XResNet)
def xse_resnext34(n_out=1000, **kwargs):
    return XResNet(SEResNeXtBlock, 1, [3, 4, 6, 3], n_out=n_out, **kwargs)

@delegates(XResNet)
def xse_resnext50(n_out=1000, **kwargs):
    return XResNet(SEResNeXtBlock, 4, [3, 4, 6, 3], n_out=n_out, **kwargs)

@delegates(XResNet)
def xse_resnext101(n_out=1000, **kwargs):
    return XResNet(SEResNeXtBlock, 4, [3, 4, 23, 3], n_out=n_out, **kwargs)

# Cell
@delegates(XResNet)
def xeca_resnet18(n_out=1000, **kwargs):
    return XResNet(ECABlock, 1, [2, 2, 2, 2], n_out=n_out, **kwargs)

@delegates(XResNet)
def xeca_resnet34(n_out=1000, **kwargs):
    return XResNet(ECABlock, 1, [3, 4, 6, 3], n_out=n_out, **kwargs)

@delegates(XResNet)
def xeca_resnet50(n_out=1000, **kwargs):
    return XResNet(ECABlock, 4, [3, 4, 6, 3], n_out=n_out, **kwargs)

@delegates(XResNet)
def xeca_resnet101(n_out=1000, **kwargs):
    return XResNet(ECABlock, 4, [3, 4, 23, 3], n_out=n_out, **kwargs)

# Cell
@delegates(XResNet)
def xeca_resnext18(n_out=1000, **kwargs):
    return XResNet(ECAResNeXtBlock, 1, [2, 2, 2, 2], n_out=n_out, **kwargs)

@delegates(XResNet)
def xeca_resnext34(n_out=1000, **kwargs):
    return XResNet(ECAResNeXtBlock, 1, [3, 4, 6, 3], n_out=n_out, **kwargs)

@delegates(XResNet)
def xeca_resnext50(n_out=1000, **kwargs):
    return XResNet(ECAResNeXtBlock, 4, [3, 4, 6, 3], n_out=n_out, **kwargs)

@delegates(XResNet)
def xeca_resnext101(n_out=1000, **kwargs):
    return XResNet(ECAResNeXtBlock, 4, [3, 4, 23, 3], n_out=n_out, **kwargs)

# Cell
@delegates(XResNet)
def xsa_resnet18(n_out=1000, **kwargs):
    return XResNet(SABlock, 1, [2, 2, 2, 2], n_out=n_out, sa_grps=32, **kwargs)

@delegates(XResNet)
def xsa_resnet34(n_out=1000, **kwargs):
    return XResNet(SABlock, 1, [3, 4, 6, 3], n_out=n_out, sa_grps=32, **kwargs)

@delegates(XResNet)
def xsa_resnet50(n_out=1000, **kwargs):
    return XResNet(SABlock, 4, [3, 4, 6, 3], n_out=n_out, **kwargs)

@delegates(XResNet)
def xsa_resnet101(n_out=1000, **kwargs):
    return XResNet(SABlock, 4, [3, 4, 23, 3], n_out=n_out, **kwargs)

# Cell
@delegates(XResNet)
def xsa_resnext18(n_out=1000, **kwargs):
    return XResNet(SAResNeXtBlock, 1, [2, 2, 2, 2], n_out=n_out, sa_grps=32, **kwargs)

@delegates(XResNet)
def xsa_resnext34(n_out=1000, **kwargs):
    return XResNet(SAResNeXtBlock, 1, [3, 4, 6, 3], n_out=n_out, sa_grps=32, **kwargs)

@delegates(XResNet)
def xsa_resnext50(n_out=1000, **kwargs):
    return XResNet(SAResNeXtBlock, 4, [3, 4, 6, 3], n_out=n_out, **kwargs)

@delegates(XResNet)
def xsa_resnext101(n_out=1000, **kwargs):
    return XResNet(SAResNeXtBlock, 4, [3, 4, 23, 3], n_out=n_out, **kwargs)

# Cell
@delegates(XResNet)
def xta_resnet18(n_out=1000, **kwargs):
    return XResNet(TABlock, 1, [2, 2, 2, 2], n_out=n_out, **kwargs)

@delegates(XResNet)
def xta_resnet34(n_out=1000, **kwargs):
    return XResNet(TABlock, 1, [3, 4, 6, 3], n_out=n_out, **kwargs)

@delegates(XResNet)
def xta_resnet50(n_out=1000, **kwargs):
    return XResNet(TABlock, 4, [3, 4, 6, 3], n_out=n_out, **kwargs)

@delegates(XResNet)
def xta_resnet101(n_out=1000, **kwargs):
    return XResNet(TABlock, 4, [3, 4, 23, 3], n_out=n_out, **kwargs)

# Cell
@delegates(XResNet)
def xta_resnext18(n_out=1000, **kwargs):
    return XResNet(TAResNeXtBlock, 1, [2, 2, 2, 2], n_out=n_out, sa_grps=32, **kwargs)

@delegates(XResNet)
def xta_resnext34(n_out=1000, **kwargs):
    return XResNet(TAResNeXtBlock, 1, [3, 4, 6, 3], n_out=n_out, sa_grps=32, **kwargs)

@delegates(XResNet)
def xta_resnext50(n_out=1000, **kwargs):
    return XResNet(TAResNeXtBlock, 4, [3, 4, 6, 3], n_out=n_out, **kwargs)

@delegates(XResNet)
def xta_resnext101(n_out=1000, **kwargs):
    return XResNet(TAResNeXtBlock, 4, [3, 4, 23, 3], n_out=n_out, **kwargs)