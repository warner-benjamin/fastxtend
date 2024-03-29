# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/0_basics.ipynb.

# %% ../nbs/0_basics.ipynb 1
# Contains code from:
# fastcore - Apache License 2.0 - Copyright (c) 2023 fast.ai

# %% ../nbs/0_basics.ipynb 3
from __future__ import annotations

import sys
import re

from fastcore.basics import annotations, argnames, _store_attr, filter_dict, range_of, Inf
from fastcore.dispatch import typedispatch, retain_meta

from .imports import *

# %% auto 0
__all__ = ['is_listish', 'listify_store_attr', 'show_batch']

# %% ../nbs/0_basics.ipynb 4
def is_listish(x):
    "Subset of `is_listy`: (tuple,list,L)"
    return isinstance(x, (tuple,list,L))

# %% ../nbs/0_basics.ipynb 5
def listify_store_attr(names=None, self=None, but='', cast=False, store_args=None, **attrs):
    "Maybe listify, then store params named in comma-separated `names` from calling context into attrs in `self`"
    fr = sys._getframe(1)
    args = argnames(fr, True)
    if self: args = ('self', *args)
    else: self = fr.f_locals[args[0]]
    if store_args is None: store_args = not hasattr(self,'__slots__')
    if store_args and not hasattr(self, '__stored_args__'): self.__stored_args__ = {}
    anno = annotations(self) if cast else {}
    if names and isinstance(names,str): names = re.split(', *', names)
    ns = names if names is not None else getattr(self, '__slots__', args[1:])
    added = {n:fr.f_locals[n] for n in ns}
    attrs = {**attrs, **added}
    if isinstance(but,str): but = re.split(', *', but)
    # start listify_store_attr code
    attrs = {k:v for k,v in attrs.items() if k not in but}
    attrs_len = {n:len(attrs[n]) if is_listy(attrs[n]) else 1 for n in attrs.keys()}
    l = max(attrs_len.values())
    if l > 1:
        ones = filter_dict(attrs_len, lambda x,v: v==1)
        if len(ones)+1 != len(attrs_len):
            raise ValueError(f'Args must be all be length {l} or 1. Invalid args: {list(filter_dict(attrs_len, lambda x,v: l>v>1).keys())}')
        for n in ones.keys():
            # if-else needed for None input
            attrs[n] = L(attrs[n])*l if is_listy(attrs[n]) else L([attrs[n]])*l
    return _store_attr(self, anno, **attrs)

# %% ../nbs/0_basics.ipynb 6
@typedispatch
def show_batch(x, y, samples, ctxs=None, max_n=9, **kwargs):
    if ctxs is None: ctxs = Inf.nones
    plots = []
    if hasattr(samples[0], 'show'):
        for s,c,_ in zip(samples,ctxs,range(max_n)):
            s = retain_meta(x, s)
            plots.append(s.show(ctx=c, **kwargs))
    else:
        for i in range_of(samples[0]):
            for b,c,_ in zip(samples.itemgot(i),ctxs,range(max_n)):
                b = retain_meta(x, b)
                plots.append(b.show(ctx=c, **kwargs))
    ctxs = plots
    return ctxs
