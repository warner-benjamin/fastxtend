
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Union, TypeVar, Callable
from fastcore.foundation import L, fastuple

T = TypeVar('T')

listy     = Union[Iterable[T], L, fastuple]
listified = Union[T, Iterable[T], L, fastuple]