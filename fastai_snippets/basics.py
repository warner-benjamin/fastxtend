# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/0_basics.ipynb (unless otherwise specified).

__all__ = ['is_listish']

# Cell
from fastcore.foundation import L

# Cell
def is_listish(x):
    "Subset of `is_listy`: (tuple,list,L)"
    return isinstance(x, (tuple,list,L))