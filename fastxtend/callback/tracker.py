# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/callback.tracker.ipynb.

# %% ../../nbs/callback.tracker.ipynb 1
# Contains code from:
# fastai - Apache License 2.0 - Copyright (c) 2023 fast.ai

# %% ../../nbs/callback.tracker.ipynb 3
from __future__ import annotations

from fastai.callback.core import Callback, CancelFitException
from fastai.callback.tracker import SaveModelCallback, TrackerCallback

from ..imports import *

# %% auto 0
__all__ = ['TerminateOnTrainNaN', 'SaveModelAtEnd', 'LastMetricCallback']

# %% ../../nbs/callback.tracker.ipynb 6
class TerminateOnTrainNaN(Callback):
    "A `Callback` that terminates training if the training loss is NaN and ignores valid loss."
    order, run_valid = -9, False
    def after_batch(self):
        "Test if `last_loss` is NaN and interrupts training."
        if torch.isinf(self.loss) or torch.isnan(self.loss): raise CancelFitException

# %% ../../nbs/callback.tracker.ipynb 9
class SaveModelAtEnd(SaveModelCallback):
    "A `SaveModelCallback` which only saves the model at the end so loggers can find it."
    order = TrackerCallback.order+1
    def __init__(self,
        fname='model', # Model filename
        with_opt=False # Include optimizer state
    ):
        # keep track of file path for loggers
        self.last_saved_path = None
        store_attr()

    def before_fit(self):
        pass

    def after_epoch(self):
        pass

    def after_fit(self, **kwargs):
        self.last_saved_path = self.learn.save(f'{self.fname}', with_opt=self.with_opt)

# %% ../../nbs/callback.tracker.ipynb 11
class LastMetricCallback(Callback):
    "A `Callback` which stores metrics by name in a `Learner.lastmetric` dictionary"
    order,remove_on_fetch,_only_train_loop = 60,True,True
    def __init__(self, metrics:Listified[str]='valid_loss'):
        self.metrics=L(metrics)

    def before_fit(self):
        "Prepare the monitored value(s)"
        self.run = not hasattr(self, "lr_finder") and not hasattr(self, "gather_preds")
        self.idx, self.learn.lastmetric = [], {}
        for m in self.metrics:
            assert m in self.recorder.metric_names[1:], f'Metric {m} does not exist'
            self.idx.append(list(self.recorder.metric_names[1:]).index(m))

    def after_fit(self):
        "Store the last the monitored value(s)"
        for i, idx in enumerate(self.idx):
            self.learn.lastmetric[self.metrics[i]] = self.recorder.values[-1][idx]
        self.run = True

    def after_fit_exception(self):
        try:
            self.after_fit()
        finally:
            self.run = True
