import math
from collections import OrderedDict, defaultdict, Counter, namedtuple
from enum import Enum
from functools import partial
from typing import Iterable, Union, TypeVar, Callable, Any
from warnings import warn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fastcore.basics import store_attr, noop, Self, hasattrs, partialler
from fastcore.foundation import L, fastuple, patch
from fastcore.meta import delegates
from fastcore.utils import ifnone
from fastcore.xtras import is_listy

from fastai.torch_core import *

T = TypeVar('T')

listy     = Union[Iterable[T], L, fastuple]
listified = Union[T, Iterable[T], L, fastuple]
Number    = Union[int, float]