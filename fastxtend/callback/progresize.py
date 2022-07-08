# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/callback.progresize.ipynb (unless otherwise specified).


from __future__ import annotations


__all__ = ['ProgressiveResize']

# Cell
#nbdev_comment from __future__ import annotations

from fastcore.basics import detuplify
from fastcore.transform import Pipeline, Transform

from fastai.callback.core import Callback
from fastai.vision.augment import AffineCoordTfm, RandomResizedCropGPU

from ..metrics import MetricX
from ..imports import *

# Internal Cell
_resize_augs = (AffineCoordTfm, RandomResizedCropGPU)

# Internal Cell
def _to_size(t:Tensor):
    "Convert Tensor to size compatible values"
    if sum(t.shape)==2: return tuple(t.tolist())
    else:               return tuple(t.item(),t.item())

# Cell
class ProgressiveResize(Callback):
    run_valid, order = False, 5 # Needs to run before MixUp et al
    "Progressively increase the size of input images during training. Final image size is the valid image size."
    def __init__(self,
        initial_size:Number|tuple[int,int]|None=0.5,
        increase_by:int|tuple[int,int]=4,
        start_at:float=0.5,
        finish_at:float=0.75,
        resize_mode:str='nearest',
        add_resize:bool=False,
        input_size:int|tuple[int,int]|None=None, # Final image size. Set if using a non-fastai Dataloaders.
        logger_callback='wandb' # Log report and samples/second to `logger_callback` using `Callback.name`
    ):
        store_attr()
        self._log_after_resize = getattr(self, f'_{self.logger_callback}_log_after_resize', noop)

    def before_fit(self):
        if hasattr(self.learn, 'lr_finder') and not hasattr(self, "gather_preds"):
            self.run = False
            return

        self.remove_resize, self.null_resize, self.remove_cutmix = True, True, False
        self.has_logger = hasattr(self.learn, self.logger_callback)

        # Try to automatically determine the input size
        try:
            n_inp = self.dls.train.n_inp
            xb = self.dls.valid.one_batch()[:n_inp]
            for n in range(n_inp):
                x = detuplify(xb[n])
                if isinstance(x, TensorImageBase):
                    self.input_size = x.shape[-2:]
        finally:
            assert self.input_size is not None, "Could not determine input size. Set `input_size` to input size."
            self.input_size = tensor(self.input_size)

        if self.input_size.equal(tensor(-1)):
            raise ValueError('No `TensorImageBase` derived inputs detected to gather shape')

        if isinstance(self.initial_size, float):
            self.current_size = (tensor(self.initial_size) * self.input_size).int()
        elif isinstance(self.initial_size, tuple):
            self.current_size = tensor(self.initial_size)

        self.increase_by = tensor(self.increase_by)
        if ((self.input_size-self.current_size) % self.increase_by).sum() != 0:
            raise ValueError(f'Resize amount {self.input_size-self.current_size} not evenly divisible by `increase_by` {self.increase_by}')

        n_steps = ((self.input_size-self.current_size) / self.increase_by).int()
        if len(n_steps.shape)==2: assert n_steps[0]==n_steps[1]
        pct = (self.finish_at - self.start_at) / (n_steps[0].item()-1)
        self.step_pcts = [self.start_at + pct*i for i in range(n_steps[0].item())]+[1.1]
        print(self.step_pcts)

        self._resize = []
         # If `add_resize`, add a seperate resize
        if self.add_resize:
            self._resize_pipe = Pipeline(AffineCoordTfm(size=_to_size(self.current_size), mode=self.resize_mode))
            self._resize.append(self._resize_pipe[0])
            self.remove_resize = True
        else:
            if hasattr(self.learn, 'cutmixupaugment'):
                # Modify the `CutMixUpAugment` augmentation pipeline
                for i in range(len(self.learn.cutmixupaugment._orig_pipe)):
                    if isinstance(self.learn.cutmixupaugment._orig_pipe[i], _resize_augs):
                        self._resize.append(self.learn.cutmixupaugment._orig_pipe[i])
                        self.null_resize = self.null_resize and self.learn.cutmixupaugment._orig_pipe[i].size is None
                        self.remove_resize = False

                # If `CutMixUpAugment` has an Affine Transform for Augmentations then
                if len(self._resize) > 0:
                    # Check for pre-mixup augment pipeline and modify it
                    if self.learn.cutmixupaugment._docutmixaug:
                        for i in range(len(self.learn.cutmixupaugment._cutmixaugs_pipe)):
                            if isinstance(self.learn.cutmixupaugment._cutmixaugs_pipe[i], _resize_augs):
                                self._resize.append(self.learn.cutmixupaugment._cutmixaugs_pipe[i])
                                self.null_resize = self.null_resize and self.learn.cutmixupaugment._cutmixaugs_pipe[i].size is None
                                self.remove_resize = False
                    else:
                        # There isn't one, then add it a pre-mixup augment pipeline for resizing
                        self.learn.cutmixupaugment._cutmixaugs_pipe = Pipeline(AffineCoordTfm(size=_to_size(self.current_size)))
                        self.learn.cutmixupaugment._docutmixaug = True
                        self._resize.append(self.learn.cutmixupaugment._cutmixaugs_pipe[0])
                        self.remove_cutmix, self.remove_resize = True, True

            else:
                # If no `CutMixUpAugment` check the dataloader pipeline for Affine Transforms
                for i in range(len(self.dls.train.after_batch.fs)):
                    if isinstance(self.dls.train.after_batch[i], _resize_augs):
                        self._resize.append(self.dls.train.after_batch[i])
                        self.null_resize = self.null_resize and self.dls.train.after_batch[i].size is None
                        self.remove_resize = False

            # If no there are no detected resizes add a resize transform pipeline
            if len(self._resize) == 0:
                self.add_resize = True
                self._resize_pipe = Pipeline(AffineCoordTfm(size=_to_size(self.current_size)))
                self._resize.append(self._resize_pipe[0])
                self.remove_resize = True

        self._orig_modes = []
        for resize in self._resize:
            resize.size = _to_size(self.current_size)
            self._orig_modes.append(resize.mode)
            resize.mode = self.resize_mode


    def before_batch(self):
        if self.add_resize:
            self.learn.xb = self._resize_pipe(self.xb)
            # self.learn.yb = self._resize_pipe(self.yb)

    def after_batch(self):
        if self.pct_train >= self.step_pcts[0]:
            self.step_pcts = self.step_pcts[1:]
            self.current_size += self.increase_by
            for i, resize in enumerate(self._resize):
                if (self.current_size < self.input_size).all():
                    resize.size = _to_size(self.current_size)
                else:
                    print("input")
                    if self.null_resize:
                        resize.size = None
                    elif self.remove_resize:
                        if self.remove_cutmix:
                            self.learn.cutmixupaugment._cutmixaugs_pipe = Pipeline([])
                            self.learn.cutmixupaugment._docutmixaug = False
                        else:
                            self._resize_pipe = Pipeline([])
                            self.add_resize = False
                    else:
                        resize.size = _to_size(self.current_size)
                        resize.mode = self._orig_modes[i]
            if self.has_logger: self._log_after_resize()

# Cell
try:
    import wandb

    @patch
    def _wandb_log_after_resize(self:ProgressiveResize):
        if len(self.current_size.shape)==2:
            size = {'progressive_resize_height': self.current_size[0],
                    'progressive_resize_width': self.current_size[1] }
        else:
            size = {'progressive_resize_size': self.current_size}
        wandb.log(size, self.learn.wandb._wandb_step+1)
except:
    pass