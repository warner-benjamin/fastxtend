# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/audio.02_data.ipynb (unless otherwise specified).


from __future__ import annotations


__all__ = ['Spectrogram', 'MelSpectrogram', 'AudioBlock', 'SpecBlock', 'MelSpecBlock']

# Cell
#nbdev_comment from __future__ import annotations

from torch.nn.functional import interpolate
import torchaudio.transforms as tatfms
import matplotlib.pyplot as plt

from fastcore.dispatch import typedispatch, explode_types
from fastcore.transform import DisplayedTransform

from fastai.vision.data import get_grid
from fastai.data.core import TfmdDL
from fastai.data.block import TransformBlock

from .core import TensorAudio, TensorSpec, TensorMelSpec
from ..imports import *
from ..basics import *

# Internal Cell
@patch
def to(self:TfmdDL, device):
    self.device = device
    for tfm in self.after_batch.fs:
        for a in L(getattr(tfm, 'parameters', None)): setattr(tfm, a, getattr(tfm, a).to(device))
        if hasattr(tfm, 'to'): tfm.to(device)
    return self

# Internal Cell
@patch
def _one_pass(self:TfmdDL):
    b = self.do_batch([self.do_item(None)])
    if self.device is not None:
        b = to_device(b, self.device)
        self.to(self.device)
    its = self.after_batch(b)
    self._n_inp = 1 if not isinstance(its, (list,tuple)) or len(its)==1 else len(its)-1
    self._types = explode_types(its)

# Internal Cell
@typedispatch
def show_batch(x:TensorAudio, y, samples, ctxs=None, max_n=9, nrows=None, ncols=None, figsize=None, **kwargs):
    if ctxs is None: ctxs = get_grid(min(len(samples), max_n), nrows=nrows, ncols=ncols, figsize=figsize)
    ctxs = show_batch[object](x, y, samples, ctxs=ctxs, max_n=max_n, hear=False, **kwargs)
    plt.tight_layout()
    return ctxs

# Internal Cell
@typedispatch
def show_batch(x:TensorSpec|TensorMelSpec, y, samples, ctxs=None, max_n=9, nrows=None, ncols=None, figsize=None, **kwargs):
    if ctxs is None: ctxs = get_grid(min(len(samples), max_n), nrows=nrows, ncols=ncols, figsize=figsize)
    ctxs = show_batch[object](x, y, samples, ctxs=ctxs, max_n=max_n, **kwargs)
    plt.tight_layout()
    return ctxs

# Cell
class Spectrogram(DisplayedTransform):
    "Convert a `TensorAudio` into one or more `TensorSpec`"
    order = 75
    def __init__(self,
        n_fft:listified[int]=1024,
        win_length:listified[int]|None=None,
        hop_length:listified[int]|None=None,
        pad:listified[int]=0,
        window_fn:listified[Callable[..., Tensor]]=torch.hann_window,
        power:listified[float]=2.,
        normalized:listified[bool]=False,
        wkwargs:listified[dict]|None=None,
        center:listified[bool]=True,
        pad_mode:listified[str]="reflect",
        onesided:listified[bool]=True,
        norm:listified[str]|None=None,
    ):
        super().__init__()
        listify_store_attr()
        attrs = {k:v for k,v in getattr(self,'__stored_args__',{}).items() if k not in ['size', 'mode']}
        # self.resize = size is not None
        if is_listy(self.n_fft):
            self.specs, self._attrs = nn.ModuleList(), []
            self.len, self.multiple = len(self.n_fft), True
            for i in range(self.len):
                self.specs.append(tatfms.Spectrogram(n_fft=self.n_fft[i], win_length=self.win_length[i],
                                                     hop_length=self.hop_length[i], pad=self.pad[i],
                                                     window_fn=self.window_fn[i], power=self.power[i],
                                                     normalized=self.normalized[i], wkwargs=self.wkwargs[i],
                                                     center=self.center[i], pad_mode=self.pad_mode[i],
                                                     onesided=self.onesided[i], norm=self.norm[i]))

                self._attrs.append({k:v[i] for k,v in self._get_attrs().items()})
        else:
            self.multiple = False
            self.spec = tatfms.Spectrogram(n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length,
                                           pad=self.pad, window_fn=self.window_fn, power=self.power,
                                           normalized=self.normalized, wkwargs=self.wkwargs, center=self.center,
                                           pad_mode=self.pad_mode, onesided=self.onesided, norm=self.norm)

            self._attrs = {k:v for k,v in self._get_attrs().items()}

    def encodes(self, x:TensorAudio):
        if self.multiple:
            specs = []
            for i in range(self.len):
                specs.append(TensorSpec.create(self.specs[i](x), settings=self._attrs[i]))
            return tuple(specs)
        else:
            return TensorSpec.create(self.spec(x), settings=self._attrs)

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if self.multiple: self.specs.to(device)
        else: self.spec.to(device)

    def _get_attrs(self):
        return {k:v for k,v in getattr(self,'__dict__',{}).items() if k in getattr(self,'__stored_args__',{}).keys()}

# Cell
class MelSpectrogram(DisplayedTransform):
    "Convert a `TensorAudio` into one or more `TensorMelSpec`"
    order = 75
    def __init__(self,
        sample_rate:listified[int]=16000,
        n_fft:listified[int]=1024,
        win_length:listified[int]|None=None,
        hop_length:listified[int]|None=None,
        f_min:listified[float]=0.,
        f_max:listified[float]|None=None,
        pad:listified[int]=0,
        n_mels:listified[int]=128,
        window_fn:listified[Callable[..., Tensor]]=torch.hann_window,
        power:listified[float]=2.,
        normalized:listified[bool]=False,
        wkwargs:listified[dict]|None=None,
        center:listified[bool]=True,
        pad_mode:listified[str]="reflect",
        onesided:listified[bool]=True,
        norm:listified[str]|None=None,
        mel_scale:listified[str]="htk",
        # size:tuple[int,int]|None=None, # If set, resize MelSpectrogram to `size`
        # mode='bilinear'
    ):
        super().__init__()
        listify_store_attr()
        # self.resize = size is not None
        if is_listy(self.n_fft):
            self.mels, self._attrs = nn.ModuleList(), []
            self.len, self.multiple = len(self.n_fft), True
            for i in range(self.len):
                self.win_length[i] = self.win_length[i] if self.win_length[i] is not None else self.n_fft[i]
                self.hop_length[i] = self.hop_length[i] if self.hop_length[i] is not None else self.win_length[i] // 2
                self.mels.append(tatfms.MelSpectrogram(sample_rate=self.sample_rate[i], n_fft=self.n_fft[i],
                                                       win_length=self.win_length[i], hop_length=self.hop_length[i],
                                                       f_min=self.f_min[i], f_max=self.f_max[i], pad=self.pad[i],
                                                       n_mels=self.n_mels[i], window_fn=self.window_fn[i], power=self.power[i],
                                                       normalized=self.normalized[i], wkwargs=self.wkwargs[i],
                                                       center=self.center[i], pad_mode=self.pad_mode[i],
                                                       onesided=self.onesided[i], norm=self.norm[i], mel_scale=self.mel_scale[i]))

                self._attrs.append({**{k:v[i] for k,v in self._get_attrs().items()},**{'sr':self.sample_rate[i]}})
        else:
            self.multiple = False
            self.win_length = self.win_length if self.win_length is not None else self.n_fft
            self.hop_length = self.hop_length if self.hop_length is not None else self.win_length // 2
            self.mel = tatfms.MelSpectrogram(sample_rate=self.sample_rate, n_fft=self.n_fft, win_length=self.win_length,
                                             hop_length=self.hop_length, f_min=self.f_min, f_max=self.f_max, pad=self.pad,
                                             n_mels=self.n_mels, window_fn=self.window_fn, power=self.power,
                                             normalized=self.normalized, wkwargs=self.wkwargs, center=self.center,
                                             pad_mode=self.pad_mode, onesided=self.onesided, norm=self.norm,
                                             mel_scale=self.mel_scale)

            self._attrs = {**{k:v for k,v in self._get_attrs().items()},**{'sr':self.sample_rate}}

    def encodes(self, x:TensorAudio):
        if self.multiple:
            mels = []
            for i in range(self.len):
                mels.append(TensorMelSpec.create(self.mels[i](x), settings=self._attrs[i]))
            return tuple(mels)
        else:
            return TensorMelSpec.create(self.mel(x), settings=self._attrs)

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if self.multiple: self.mels.to(device)
        else: self.mel.to(device)

    def _get_attrs(self):
        return {k:v for k,v in getattr(self,'__dict__',{}).items() if k in getattr(self,'__stored_args__',{}).keys() and k not in ['size', 'mode']}

# Cell
def AudioBlock(cls=TensorAudio):
    "A `TransformBlock` for audio of `cls`"
    return TransformBlock(type_tfms=cls.create)

# Cell
def SpecBlock(cls=TensorAudio,
    # Spectrogram args
    n_fft:listified[int]=1024,
    win_length:listified[int]|None=None,
    hop_length:listified[int]|None=None,
    pad:listified[int]=0,
    window_fn:listified[Callable[..., Tensor]]=torch.hann_window,
    power:listified[float]=2.,
    normalized:listified[bool]=False,
    wkwargs:listified[dict]|None=None,
    center:listified[bool]=True,
    pad_mode:listified[str]="reflect",
    onesided:listified[bool]=True,
    norm:listified[str]|None=None
):
    "A `TransformBlock` to read `TensorAudio` and then use the GPU to turn audio into one or more `Spectrogram`s"
    return TransformBlock(type_tfms=cls.create,
                          batch_tfms=[Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                                                  pad=pad, window_fn=window_fn, power=power, normalized=normalized,
                                                  wkwargs=wkwargs, center=center, pad_mode=pad_mode,
                                                  onesided=onesided, norm=norm)])

# Cell
def MelSpecBlock(cls=TensorAudio,
    # MelSpectrogram args
    sr:listified[int]=16000,
    n_fft:listified[int]=1024,
    win_length:listified[int]|None=None,
    hop_length:listified[int]|None=None,
    f_min:listified[float]=0.,
    f_max:listified[float]|None=None,
    pad:listified[int]=0,
    n_mels:listified[int]=128,
    window_fn:listified[Callable[..., Tensor]]=torch.hann_window,
    power:listified[float]=2.,
    normalized:listified[bool]=False,
    wkwargs:listified[dict]|None=None,
    center:listified[bool]=True,
    pad_mode:listified[str]="reflect",
    onesided:listified[bool]=True,
    norm:listified[str]|None=None,
    mel_scale:listified[str]="htk"
):
    "A `TransformBlock` to read `TensorAudio` and then use the GPU to turn audio into one or more `MelSpectrogram`s"
    return TransformBlock(type_tfms=cls.create,
                          batch_tfms=[MelSpectrogram(sample_rate=sr, n_fft=n_fft, win_length=win_length,
                                                     hop_length=hop_length, f_min=f_min, f_max=f_max, pad=pad,
                                                     n_mels=n_mels, window_fn=window_fn, power=power,
                                                     normalized=normalized, wkwargs=wkwargs, center=center,
                                                     pad_mode=pad_mode, onesided=onesided, norm=norm,
                                                     mel_scale=mel_scale)])