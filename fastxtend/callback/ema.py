# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/callback.ema.ipynb (unless otherwise specified).

__all__ = ['EMACallback']

# Cell
try:
    import timm
except ImportError:
    raise ImportError("timm is required to use EMACallback. Install via `pip install timm`.")

# Cell
import torch
from timm.utils.model_ema import ModelEmaV2
from fastai.callback.core import Callback
from fastcore.basics import store_attr

# Cell
class EMACallback(Callback):
    run_valid = False
    "Callback to implment Model Exponential Moving Average from PyTorch Image Models in fast.ai"
    def __init__(self, decay=0.9998, ema_device=None):
        store_attr()

    @torch.no_grad()
    def before_fit(self):
        self.ema_model = ModelEmaV2(self.learn.model, self.decay, self.ema_device)

    def after_batch(self):
        self.ema_model.update(self.learn.model)

    def before_validate(self):
        self.temp_model = self.learn.model
        self.learn.model = self.ema_model.module

    def after_validate(self):
        self.learn.model = self.temp_model

    @torch.no_grad()
    def after_fit(self):
        self.learn.model = self.ema_model.module
        self.ema_model = None
        self.remove_cb(EMACallback)