# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/data.transforms.ipynb.

# %% ../../nbs/data.transforms.ipynb 1
# Contains code from:
# fastai - Apache License 2.0 - Copyright (c) 2023 fast.ai

# %% ../../nbs/data.transforms.ipynb 3
from __future__ import annotations

import pandas as pd
from pathlib import Path

from fastcore.foundation import mask2idxs

from fastai.data.transforms import IndexSplitter

from ..imports import *

# %% auto 0
__all__ = ['KFoldColSplitter', 'ParentSplitter', 'GreatGrandparentSplitter']

# %% ../../nbs/data.transforms.ipynb 6
def KFoldColSplitter(
    fold:Listified[int]=0, # Valid set fold(s)
    col:int|str='folds' # Column with folds
):
    "Split `items` (supposed to be a dataframe) by `fold` in `col`"
    def _inner(o):
        assert isinstance(o, pd.DataFrame), "KFoldColSplitter only works when your items are a pandas DataFrame"
        valid_col = o.iloc[:,col] if isinstance(col, int) else o[col]
        valid_idx = valid_col.isin(fold) if is_listy(fold) else valid_col.values == fold
        return IndexSplitter(mask2idxs(valid_idx))(o)
    return _inner

# %% ../../nbs/data.transforms.ipynb 10
def _parent_idxs(items, name):
    def _inner(items, name): return mask2idxs(Path(o).parent.name == name for o in items)
    return [i for n in L(name) for i in _inner(items,n)]

# %% ../../nbs/data.transforms.ipynb 11
def ParentSplitter(
    train_name:str='train', # Train set folder name
    valid_name:str='valid' # Valid set folder name
):
    "Split `items` from the parent folder names (`train_name` and `valid_name`)."
    def _inner(o):
        return _parent_idxs(o, train_name),_parent_idxs(o, valid_name)
    return _inner

# %% ../../nbs/data.transforms.ipynb 14
def _greatgrandparent_idxs(items, name):
    def _inner(items, name): return mask2idxs(Path(o).parent.parent.parent.name == name for o in items)
    return [i for n in L(name) for i in _inner(items,n)]

# %% ../../nbs/data.transforms.ipynb 15
def GreatGrandparentSplitter(
    train_name:str='train', # Train set folder name
    valid_name:str='valid' # Valid set folder name
):
    "Split `items` from the great grand parent folder names (`train_name` and `valid_name`)."
    def _inner(o):
        return _greatgrandparent_idxs(o, train_name),_greatgrandparent_idxs(o, valid_name)
    return _inner
