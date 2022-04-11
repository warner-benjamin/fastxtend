import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Iterable, Union, TypeVar, Callable
from warnings import warn
from fastcore.basics import store_attr, noop
from fastcore.foundation import L, fastuple, patch
from fastcore.meta import delegates
from fastcore.utils import ifnone

T = TypeVar('T')

listy     = Union[Iterable[T], L, fastuple]
listified = Union[T, Iterable[T], L, fastuple]
Number    = Union[int, float]