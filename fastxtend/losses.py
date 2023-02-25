# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/losses.ipynb.

# %% ../nbs/losses.ipynb 2
from __future__ import annotations

import torch.nn.modules.loss as TL

from .imports import *

# %% auto 0
__all__ = ['BCEWithLogitsLoss', 'ClassBalancedCrossEntropyLoss', 'ClassBalancedBCEWithLogitsLoss']

# %% ../nbs/losses.ipynb 4
class BCEWithLogitsLoss(TL._Loss):
    """
    Like `nn.BCEWithLogitsLoss`, but with 'batchmean' reduction from MosiacML. `batchmean` scales
    loss by the batch size which results in larger loss values more similar to `nn.CrossEntropy`
    then `mean` reduction.
    """
    def __init__(self,
        weight:Tensor|None=None, # Rescaling weight for each class
        reduction:str='mean', # Pytorch reduction to apply to loss output. Also supports 'batchmean'.
        pos_weight:Tensor|None=None, # Weight of positive examples
        thresh:float=0.5, # Threshold for `decodes`
    ):
        super().__init__(None, None, reduction)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)
        self.weight:Tensor|None
        self.pos_weight:Tensor|None
        self.thresh = thresh

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.reduction == 'batchmean':
            return F.binary_cross_entropy_with_logits(input, target,
                                                      self.weight,
                                                      pos_weight=self.pos_weight,
                                                      reduction='sum')/torch.tensor(input.shape[0])
        else:
            return F.binary_cross_entropy_with_logits(input, target,
                                                      self.weight,
                                                      pos_weight=self.pos_weight,
                                                      reduction=self.reduction)

    def decodes(self, x:Tensor) -> Tensor:
        "Converts model output to target format"
        return x > self.thresh

    def activation(self, x:Tensor) -> Tensor:
        "`nn.BCEWithLogitsLoss`'s fused activation function applied to model output"
        return torch.sigmoid(x)

# %% ../nbs/losses.ipynb 6
# Refactored from https://github.com/vandit15/Class-balanced-loss-pytorch
# Class-balanced-loss-pytorch - MIT License - Copyright (c) 2019 Vandit Jain

class ClassBalanced(TL._WeightedLoss):
    "Class Balanced weight calculation, from https://arxiv.org/abs/1901.05555."
    def __init__(self,
        samples_per_class:Tensor|Listy[int], # Number of samples per class
        beta:float=0.99, # Rebalance factor, usually between [0.9, 0.9999]
        reduction:str='mean', # Pytorch reduction to apply to loss output
    ):
        num_classes = len(samples_per_class)
        if not isinstance(samples_per_class, Tensor):
            samples_per_class = Tensor(samples_per_class)

        effective_num = 1.0 - torch.pow(beta, samples_per_class)
        weight = (1.0 - beta) / effective_num
        weight = weight / weight.sum() * num_classes

        super().__init__(weight, None, None, reduction)

    @delegates(nn.Module.to)
    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if self.weight.device != device: self.weight = self.weight.to(device=device)
        super().to(*args, **kwargs)

# %% ../nbs/losses.ipynb 7
class ClassBalancedCrossEntropyLoss(ClassBalanced):
    "Class Balanced Cross Entropy Loss, from https://arxiv.org/abs/1901.05555."
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(self,
        samples_per_class:Tensor|Listy[int], # Number of samples per class
        beta:float=0.99, # Rebalance factor, usually between [0.9, 0.9999]
        ignore_index:int=-100, # Target value which is ignored and doesn't contribute to gradient
        reduction:str='mean', # Pytorch reduction to apply to loss output
        label_smoothing:float=0.0, # Convert hard targets to soft targets, defaults to no smoothing
        axis:int=-1 # ArgMax axis for fastai `decodes``
    ):
        super().__init__(samples_per_class, beta, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.axis = axis

    def forward(self, input:Tensor, target:Tensor) -> Tensor:
        return F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction,
                               label_smoothing=self.label_smoothing)

    def decodes(self, x:Tensor):
        "Converts model output to target format"
        return x.argmax(dim=self.axis)

    def activation(self, x:Tensor):
        "`nn.CrossEntropyLoss`'s fused activation function applied to model output"
        return F.softmax(x, dim=self.axis)

# %% ../nbs/losses.ipynb 8
class ClassBalancedBCEWithLogitsLoss(ClassBalanced):
    "Class Balanced BCE With Logits Loss, from https://arxiv.org/abs/1901.05555 with 'batchmean' reduction"

    def __init__(self,
        samples_per_class:Tensor|Listy[int], # Number of samples per class
        beta:float=0.99, # Rebalance factor, usually between [0.9, 0.9999]
        reduction:str='mean', # Pytorch reduction to apply to loss output. Also supports 'batchmean'.
        pos_weight:Tensor|None=None, # BCE Weight of positive examples
        thresh:float=0.5, # Threshold for fastai `decodes`
    ):
        super().__init__(samples_per_class, beta, reduction)
        self.register_buffer('pos_weight', pos_weight)
        self.pos_weight:Tensor|None
        self.thresh = thresh

    def forward(self, input:Tensor, target:Tensor) -> Tensor:
        if self.reduction == 'batchmean':
            return F.binary_cross_entropy_with_logits(input, target,
                                                      self.weight,
                                                      pos_weight=self.pos_weight,
                                                      reduction='sum')/torch.tensor(input.shape[0])
        else:
            return F.binary_cross_entropy_with_logits(input, target,
                                                      self.weight,
                                                      pos_weight=self.pos_weight,
                                                      reduction=self.reduction)

    def decodes(self, x:Tensor) -> Tensor:
        "Converts model output to target format"
        return x > self.thresh

    def activation(self, x:Tensor) -> Tensor:
        "`nn.BCEWithLogitsLoss`'s fused activation function applied to model output"
        return torch.sigmoid(x)
