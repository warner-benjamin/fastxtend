# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/callback.progresize.ipynb (unless otherwise specified).


from __future__ import annotations


__all__ = ['ProgSizeMode', 'ProgressiveResize']

# Cell
#nbdev_comment from __future__ import annotations

from fastcore.basics import detuplify
from fastcore.transform import Pipeline, Transform

from fastai.callback.core import Callback
from fastai.vision.augment import AffineCoordTfm, RandomResizedCropGPU

from .cutmixup import CutMixUpAugment
from ..imports import *

# Internal Cell
_resize_augs = (AffineCoordTfm, RandomResizedCropGPU)

# Internal Cell
def _to_size(t:Tensor):
    "Convert Tensor to size compatible values"
    if sum(t.shape)==2: return tuple(t.tolist())
    else:               return tuple(t.item(),t.item())

# Internal Cell
def _num_steps(input_size, current_size, min_increase):
    "Convert Tensor to size compatible values"
    steps = (input_size - current_size) / min_increase
    if sum(steps.shape)==2:
        steps = steps[0].item()
    return steps

# Internal Cell
def _evenly_divisible(input_size, current_size, min_increase, steps):
    min_increase = tensor(min_increase)
    return (((input_size-current_size) % min_increase).sum() == 0) and (((input_size-current_size) - (min_increase*steps)).sum() == 0)

# Cell
class ProgSizeMode(Enum):
    "Delete batch after resize to assist with PyTorch memory management"
    Auto = 'auto'
    Strict = 'strict'

# Cell
class ProgressiveResize(Callback):
    order = CutMixUpAugment.order+1 # Needs to run after CutMixUpAugment
    "Progressively increase the size of input images during training. Final image size is the valid image size or `input_size`."
    def __init__(self,
        initial_size:float|tuple[int,int]=0.5, # Staring size to increase from. Image shape must be square
        start:Number=0.5, # Earliest upsizing epoch in percent of training time or epoch (index 0)
        finish:Number=0.75, # Last upsizing epoch in percent of training time or epoch (index 0)
        min_increase:int=4, # Minimum increase per resising epoch
        size_mode:ProgSizeMode=ProgSizeMode.Auto, # Automatically determine the resizing schedule
        resize_mode:str='bilinear', # PyTorch interpolate mode string for progressive resizing
        add_resize:bool=False, # Add a seperate resize step. Use if for non-fastai DataLoaders or DataLoaders without batch transforms
        resize_valid:bool=True, # Apply progressive resizing to valid dataset
        input_size:tuple[int,int]|None=None, # Final image size. Set if using a non-fastai DataLoaders.
        logger_callback:str='wandb', # Log report and samples/second to `logger_callback` using `Callback.name` if avalible
        empty_cache:bool=True, # Call `torch.cuda.empty_cache()` after each epoch to prevent memory allocation overflow
        verbose:str=True, # Print a summary of the progressive resizing schedule
    ):
        store_attr()
        self.run_valid = resize_valid

    def before_fit(self):
        if hasattr(self.learn, 'lr_finder') and not hasattr(self, "gather_preds"):
            self.run = False
            return

        self.remove_resize, self.null_resize, self.remove_cutmix = True, True, False
        self._log_after_resize = getattr(self, f'_{self.logger_callback}_log_after_resize', noop)
        self.has_logger = hasattr(self.learn, self.logger_callback) and self._log_after_resize != noop
        self.min_increase = tensor(self.min_increase)

        # Try to automatically determine the input size
        try:
            n_inp = self.dls.train.n_inp
            xb = self.dls.valid.one_batch()[:n_inp]
            for n in range(n_inp):
                x = detuplify(xb[n])
                if isinstance(x, TensorImageBase):
                    self.input_size = x.shape[-2:]
        finally:
            if self.input_size is None:
                raise ValueError(f'Could not determine input size. Set `input_size`: {self.input_size}')
            self.input_size = tensor(self.input_size)
            if self.input_size[0] != self.input_size[1]:
                raise ValueError(f'`ProgressiveResize` does not support non-square images: `input_size` = {self.input_size.tolist()}')
            if self.input_size[0] % 2 != 0:
                 raise ValueError(f"Input shape must be even: {self.input_size}")
            assert self.min_increase.item() % 2 == 0, f"Minimum increase must be even: {self.min_increase}"

        # Set the initial resize
        if isinstance(self.initial_size, float):
            self.current_size = (tensor(self.initial_size) * self.input_size).int()
        elif isinstance(self.initial_size, tuple):
            self.current_size = tensor(self.initial_size)

        start_epoch  = int(self.n_epoch*self.start)  if self.start < 1  else self.start
        finish_epoch = int(self.n_epoch*self.finish) if self.finish < 1 else self.finish
        max_steps = finish_epoch - start_epoch

        # Automatically determine the number of steps, increasing `min_increase` as needed
        if self.size_mode == ProgSizeMode.Auto:
            count = 10000 # prevent infinite loop
            steps = _num_steps(self.input_size, self.current_size, self.min_increase)
            while ((steps > max_steps) or not _evenly_divisible(self.input_size, self.current_size, self.min_increase, steps)) and count > 0:
                self.min_increase += 2
                steps = _num_steps(self.input_size, self.current_size, self.min_increase)
                count -= 1
        n_steps = _num_steps(self.input_size, self.current_size, self.min_increase)

        # Double check that the number of resize steps works
        if (n_steps > max_steps) or ((max_steps % n_steps != 0) and self.size_mode != ProgSizeMode.Auto):
            raise ValueError(f'invalid number of steps {n_steps}')

        # Double check that the step size works
        if not _evenly_divisible(self.input_size, self.current_size, self.min_increase, n_steps):
            raise ValueError(f'Resize amount {self.input_size-self.current_size} not evenly divisible by `min_increase` {self.min_increase}')

        # Set when progressive resizing steps are applied
        step_size = int(max_steps / n_steps)
        start_epoch = finish_epoch - ((self.input_size-self.current_size) / self.min_increase)*step_size
        if isinstance(start_epoch, torch.Tensor):
            if sum(start_epoch.shape)==2: start_epoch = int(start_epoch[0].item())
            else:                         start_epoch = int(start_epoch.item())
        self.step_epochs = [i for i in range(start_epoch+step_size, finish_epoch+step_size, step_size)]

        if self.verbose:
            msg = f'Progressively increase the initial image size of {self.current_size.tolist()} by {self.min_increase} '\
                  f'pixels every {step_size} epoch{"s" if step_size > 1 else ""} for {len(self.step_epochs)} resizes.\nStarting '\
                  f'at epoch {start_epoch+step_size} and finishing at epoch {finish_epoch} for a final training size of '\
                  f'{(self.current_size+(len(self.step_epochs))*self.min_increase).tolist()}.'
            print(msg)

        self._resize = []
        # If `add_resize`, add a seperate resize
        if self.add_resize:
            self._resize_pipe = Pipeline(AffineCoordTfm(size=_to_size(self.current_size), mode=self.resize_mode))
            self._resize.append(self._resize_pipe[0])
            self.remove_resize = True
        else:
            if hasattr(self.learn, 'cutmixupaugment'):
                # Modify the `CutMixUpAugment` augmentation pipeline
                self._process_pipeline(self.learn.cutmixupaugment._orig_pipe, False)

                # If `CutMixUpAugment` has an Affine Transform for Augmentations then use it
                if len(self._resize) > 0:
                    # Check for pre-mixup augment pipeline and modify it
                    if self.learn.cutmixupaugment._docutmixaug:
                        self._process_pipeline(self.learn.cutmixupaugment._cutmixaugs_pipe, False)
                    else:
                        # There isn't one, then add it a pre-mixup augment pipeline for resizing
                        self.learn.cutmixupaugment._cutmixaugs_pipe = Pipeline(AffineCoordTfm(size=_to_size(self.current_size)))
                        self.learn.cutmixupaugment._docutmixaug = True
                        self._resize.append(self.learn.cutmixupaugment._cutmixaugs_pipe[0])
                        self.remove_cutmix, self.remove_resize = True, True

            else:
                # If no `CutMixUpAugment` check the train dataloader pipeline for Affine Transforms
                self._process_pipeline(self.dls.train.after_batch.fs, False)

            # If `resize_valid` check the valid dataloader pipeline for Affine Transforms
            if self.resize_valid:
                self._process_pipeline(self.dls.valid.after_batch.fs, False)

            # If no there are no detected resizes add a resize transform pipeline
            if len(self._resize) == 0:
                self.add_resize = True
                self._resize_pipe = Pipeline(AffineCoordTfm(size=_to_size(self.current_size)))
                self._resize.append(self._resize_pipe[0])
                self.remove_resize = True

        # Set created or detected resize to the first size and store original interpolation
        self._orig_modes = []
        for resize in self._resize:
            resize.size = _to_size(self.current_size)
            self._orig_modes.append(resize.mode)
            resize.mode = self.resize_mode


    def before_batch(self):
        if self.add_resize:
            self.learn.xb = self._resize_pipe(self.xb)
            # self.learn.yb = self._resize_pipe(self.yb) TODO this wasn't working

    def before_train(self):
        if len(self.step_epochs)> 0 and self.epoch >= self.step_epochs[0]:
            _ = self.step_epochs.pop(0)
            self.current_size += self.min_increase

            for i, resize in enumerate(self._resize):
                if (self.current_size < self.input_size).all():
                    resize.size = _to_size(self.current_size)
                else:
                    # Reset everything after progressive resizing is done
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

    def after_epoch(self):
        # This hacky fix prevents fastai/PyTorch from an exploding allocation of GPU RAM which can cause training to slowdown
        if self.empty_cache and len(self.step_epochs) > 0 and self.epoch+1 >= self.step_epochs[0]:
            del self.learn.xb
            del self.learn.yb
            del self.learn.pred
            torch.cuda.empty_cache()

    def _process_pipeline(self, pipe, remove_resize=False, null_resize=None):
        for p in pipe:
            if isinstance(p, _resize_augs):
                self._resize.append(p)
                if null_resize is None:
                    self.null_resize = self.null_resize and p.size is None
                else:
                    self.null_resize = null_resize
        self.remove_resize = remove_resize

# Cell
try:
    import wandb

    @patch
    def _wandb_log_after_resize(self:ProgressiveResize):
        size = _to_size(self.current_size)
        wandb.log({'progressive_resize_size': size[0]}, self.learn.wandb._wandb_step+1)
except:
    pass