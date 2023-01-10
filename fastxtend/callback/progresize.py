# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/callback.progresize.ipynb.

# %% ../../nbs/callback.progresize.ipynb 3
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from fastcore.basics import detuplify
from fastcore.transform import Pipeline

from fastai.callback.core import Callback
from fastai.callback.fp16 import MixedPrecision
from fastai.learner import _cast_tensor
from fastai.vision.augment import AffineCoordTfm, RandomResizedCropGPU

from ..imports import *

# %% auto 0
__all__ = ['IncreaseMode', 'ProgressiveResize']

# %% ../../nbs/callback.progresize.ipynb 4
_resize_augs = (AffineCoordTfm, RandomResizedCropGPU)

# %% ../../nbs/callback.progresize.ipynb 5
def _to_size(t:Tensor):
    "Convert Tensor to size compatible values"
    if sum(t.shape)==2:
        return tuple(t.tolist())
    else:
        return tuple(t.item(),t.item())

# %% ../../nbs/callback.progresize.ipynb 6
def _num_steps(final_size, current_size, increase_by):
    "Convert Tensor to size compatible values"
    steps = (final_size - current_size) / increase_by
    if sum(steps.shape)==2:
        steps = steps[0].item()
    return steps

# %% ../../nbs/callback.progresize.ipynb 7
def _evenly_divisible(final_size, current_size, increase_by, steps):
    increase_by = tensor(increase_by)
    return (((final_size-current_size) % increase_by).sum() == 0) and (((final_size-current_size) - (increase_by*steps)).sum() == 0)

# %% ../../nbs/callback.progresize.ipynb 8
class IncreaseMode(Enum):
    "Increase mode for `ProgressiveResize`"
    Epoch = 'epoch'
    Batch = 'batch'

# %% ../../nbs/callback.progresize.ipynb 9
class ProgressiveResize(Callback):
    order = MixedPrecision.order+1 # Needs to run after MixedPrecision
    "Progressively increase the size of input images during training. Starting from `initial_size` and ending at the valid image size or `final_size`."
    def __init__(self,
        initial_size:float|tuple[int,int]=0.5, # Staring size to increase from. Image shape must be square
        start:Numeric=0.5, # Earliest upsizing epoch in percent of training time or epoch (index 0)
        finish:Numeric=0.75, # Last upsizing epoch in percent of training time or epoch (index 0)
        increase_by:int=4, # Progressively increase image size by `increase_by`, or minimum increase per upsizing epoch
        increase_mode:IncreaseMode=IncreaseMode.Batch, # Increase image size by training percent or before an epoch starts
        resize_mode:str='bilinear', # PyTorch interpolate mode string for upsizing. Resets to existing fastai DataLoader mode at `final_size`.
        resize_valid:bool=True, # Apply progressive resizing to valid dataset
        final_size:tuple[int,int]|None=None, # Final image size. Set if using a non-fastai DataLoaders, automatically detected from fastai DataLoader with batch_tfms
        add_resize:bool=False, # Add a separate resize step. Use for non-fastai DataLoaders or fastai DataLoader without batch_tfms
        resize_targ:bool=False, # Applies the separate resize step to targets
        empty_cache:bool=False, # Call `torch.cuda.empty_cache()` before a resizing epoch. May prevent Cuda & Magma errors. Don't use with multiple GPUs
        verbose:str=True, # Print a summary of the progressive resizing schedule
        logger_callback:str='wandb', # Log image size to `logger_callback` using `Callback.name` if available
    ):
        store_attr()
        self.run_valid = resize_valid
        if resize_targ and not add_resize:
            warn(f'`resize_targ` requires `add_resize` set to True')
        if empty_cache and increase_mode==IncreaseMode.Batch:
            warn(f'`empty_cache` requires `increase_mode` set to Epoch')

    def before_fit(self):
        "Sets up Progressive Resizing"
        if hasattr(self.learn, 'lr_finder') or hasattr(self.learn, "gather_preds"):
            self.run = False
            return

        self._resize, self.remove_resize, self.null_resize, self.remove_cutmix = [], True, True, False
        self._log_size = getattr(self, f'_{self.logger_callback}_log_size', noop)
        self.has_logger = hasattr(self.learn, self.logger_callback) and self._log_size != noop
        self.increase_by = tensor(self.increase_by)
        self.resize_batch = self.increase_mode == IncreaseMode.Batch

        # Dry run at full resolution to pre-allocate memory
        # See https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#pre-allocate-memory-in-case-of-variable-input-length
        states = get_random_states()
        path = self.path/self.model_dir
        path.mkdir(parents=True, exist_ok=True)
        tmp_d = TemporaryDirectory(dir=path)
        tmp_p = Path(tmp_d.name).stem
        self.learn.save(f'{tmp_p}/_tmp')
        try:
            b = self.dls.valid.one_batch()
            i = getattr(self.dls, 'n_inp', 1 if len(b)==1 else len(b)-1)
            self.learn.xb, self.learn.yb = b[:i],b[i:]

            if hasattr(self.learn, 'mixed_precision'):
                self.learn.mixed_precision.autocast.__enter__()

            self.learn.pred = self.learn.model(*_cast_tensor(self.learn.xb))
            self.learn.loss = self.learn.loss_func(self.learn.pred, *_cast_tensor(self.learn.yb))

            if hasattr(self.learn, 'mixed_precision'):
                self.learn.mixed_precision.autocast.__exit__(None, None, None)

            self.learn.loss.backward()
            self.learn.opt.zero_grad()

        finally:
            self.learn.load(f'{tmp_p}/_tmp', with_opt=True)
            tmp_d.cleanup()
            set_random_states(**states)

        # Try to automatically determine the input size
        try:
            for n in range(i):
                x = detuplify(self.learn.xb[n])
                if isinstance(x, TensorImageBase):
                    self.final_size = x.shape[-2:]
        finally:
            if self.final_size is None:
                raise ValueError(f'Could not determine image size from DataLoader. Set `final_size`: {self.final_size}')
            self.final_size = tensor(self.final_size)
            if self.final_size[0] != self.final_size[1]:
                raise ValueError(f'`ProgressiveResize` does not support non-square images: `final_size` = {self.final_size.tolist()}')
            if not self.resize_batch:
                if self.final_size[0] % 2 != 0:
                    raise ValueError(f"In Epoch mode, input image size must be even: {self.final_size.tolist()}")
                if self.increase_by.item() % 2 != 0:
                    raise ValueError(f"In Epoch Mode, `increase_by` must be even: {self.increase_by}")

        # Set the initial size
        if isinstance(self.initial_size, float):
            self.current_size = (tensor(self.initial_size) * self.final_size).int()
        elif isinstance(self.initial_size, tuple):
            self.current_size = tensor(self.initial_size)

        if self.resize_batch:
            # Set when the progressive resizing step is applied in training percent
            self.start  = self.start/self.n_epoch  if isinstance(self.start, int)  else self.start
            self.finish = self.finish/self.n_epoch if isinstance(self.finish, int) else self.finish
            n_steps = ((self.final_size-self.current_size) / self.increase_by).int()
            if sum(n_steps.shape)==2:
                n_steps = n_steps[0].item()
            pct = (self.finish - self.start) / (n_steps-1)
            self.step_pcts = [self.start + pct*i for i in range(n_steps)]
        else:
            # Automatically determine the number of steps, increasing `increase_by` as needed
            start_epoch  = int(self.n_epoch*self.start)  if isinstance(self.start, float)  else self.start
            finish_epoch = int(self.n_epoch*self.finish) if isinstance(self.finish, float) else self.finish
            max_steps = finish_epoch - start_epoch
            count = 10000 # prevent infinite loop
            steps = _num_steps(self.final_size, self.current_size, self.increase_by)
            while ((steps > max_steps) or not _evenly_divisible(self.final_size, self.current_size, self.increase_by, steps)) and count > 0:
                self.increase_by += 2
                steps = _num_steps(self.final_size, self.current_size, self.increase_by)
                count -= 1
            n_steps = _num_steps(self.final_size, self.current_size, self.increase_by)

            # Set when per epoch progressive resizing steps are applied
            step_size = int(max_steps / n_steps)
            start_epoch = finish_epoch - ((self.final_size-self.current_size) / self.increase_by)*step_size
            if isinstance(start_epoch, torch.Tensor):
                if sum(start_epoch.shape)==2: start_epoch = int(start_epoch[0].item())
                else:                         start_epoch = int(start_epoch.item())
            self.step_epochs = [i for i in range(start_epoch+step_size, finish_epoch+step_size, step_size)]


        # Double check that the step size works
        if not _evenly_divisible(self.final_size, self.current_size, self.increase_by, n_steps):
            raise ValueError(f'Resize amount {self.final_size-self.current_size} not evenly divisible by `increase_by` {self.increase_by}')

        if self.verbose:
            if self.resize_batch:
                msg = f'Progressively increase the initial image size of {self.current_size.tolist()} by {self.increase_by} '\
                      f'pixels every {pct*self.n_epoch:.4g} epochs for {len(self.step_pcts)} resizes. \nStarting at epoch '\
                      f'{self.step_pcts[0]*self.n_epoch:.4g} and finishing at epoch {self.step_pcts[-1]*self.n_epoch:.4g} '\
                      f'for a final training size of {(self.current_size+(len(self.step_pcts))*self.increase_by).tolist()}.'
                print(msg)
            else:
                msg = f'Progressively increase the initial image size of {self.current_size.tolist()} by {self.increase_by} '\
                      f'pixels every {step_size} epoch{"s" if step_size > 1 else ""} for {len(self.step_epochs)} resizes.\nStarting '\
                      f'at epoch {start_epoch+step_size} and finishing at epoch {finish_epoch} for a final training size of '\
                      f'{(self.current_size+(len(self.step_epochs))*self.increase_by).tolist()}.'
                print(msg)

        # If not `add_resize`, check for fastai Augmentation resizes to use
        if not self.add_resize:
            if hasattr(self.learn, 'cut_mix_up_augment'):
                self._has_cutmixupaug = True
                # Modify the `CutMixUpAugment` augmentation pipeline
                self._process_pipeline(self.learn.cut_mix_up_augment._orig_pipe, False)

                # If `CutMixUpAugment` has an Affine Transform for Augmentations then use it
                if len(self._resize) > 0:
                    # Check for pre-mixup augment pipeline and modify it
                    if self.learn.cut_mix_up_augment._docutmixaug:
                        self._process_pipeline(self.learn.cut_mix_up_augment._cutmixaugs_pipe, False)
                        self.learn.cut_mix_up_augment._size = _to_size(self.current_size)
                    else:
                        # There isn't one, then add it a pre-mixup augment pipeline for resizing
                        self.learn.cut_mix_up_augment._cutmixaugs_pipe = Pipeline(AffineCoordTfm(size=_to_size(self.current_size)))
                        self.learn.cut_mix_up_augment._docutmixaug = True
                        self.learn.cut_mix_up_augment._size = _to_size(self.current_size)
                        self._resize.append(self.learn.cut_mix_up_augment._cutmixaugs_pipe[0])
                        self.remove_cutmix, self.remove_resize = True, True
            else:
                self._has_cutmixupaug = False
                # If no `CutMixUpAugment` check the train dataloader pipeline for Affine Transforms
                self._process_pipeline(self.dls.train.after_batch.fs, False)

            # If `resize_valid` check the valid dataloader pipeline for Affine Transforms
            if self.resize_valid:
                self._process_pipeline(self.dls.valid.after_batch.fs, False)

        # If `add_resize` or missing a fastai Augmentation resize add a seperate resize
        if self.add_resize or len(self._resize) == 0:
            self._added_resize = partial(F.interpolate, mode=self.resize_mode, recompute_scale_factor=True)
            self.add_resize, self.remove_resize = True, True

        # Set created or detected resize to the first size and store original interpolation
        self._orig_modes = []
        for resize in self._resize:
            resize.size = _to_size(self.current_size)
            self._orig_modes.append(resize.mode)
            resize.mode = self.resize_mode

    def before_batch(self):
        "Increases the image size before a batch if set to ProgSizeMode.Batch and applies optional additional resize"
        if self.training and self.resize_batch and len(self.step_pcts) > 0 and self.pct_train >= self.step_pcts[0]:
            _ = self.step_pcts.pop(0)
            self._increase_size()
        if self.add_resize:
            self.learn.xb = (self._added_resize(self.x, scale_factor=(self.current_size/self.final_size)[0]),)
            if self.resize_targ:
                self.learn.yb = (self._added_resize(self.y, scale_factor=(self.current_size/self.final_size)[0]),)

    def before_train(self):
        "Increases the image size before the training epoch if set to ProgSizeMode.Epoch"
        if self.epoch==0 and self.has_logger:
            self._log_size(False)

        if not self.resize_batch and len(self.step_epochs) > 0 and self.epoch >= self.step_epochs[0]:
            _ = self.step_epochs.pop(0)
            self._increase_size()

    def after_epoch(self):
        "Calls `torch.cuda.empty_cache()` if `empty_cache=True` before a resizing epoch if set to ProgSizeMode.Epoch. May slightly increase single GPU training speed."
        if not self.resize_batch and self.empty_cache and len(self.step_epochs) > 0 and self.epoch+1 >= self.step_epochs[0]:
            del self.learn.xb
            del self.learn.yb
            del self.learn.pred
            torch.cuda.empty_cache()

        if self.epoch+1==self.n_epoch and self.has_logger:
            self._log_size(False)

    def _increase_size(self):
        "Increase the input size"
        if self.has_logger:
            self._log_size(False)

        self.current_size += self.increase_by
        for i, resize in enumerate(self._resize):
            if (self.current_size < self.final_size).all():
                resize.size = _to_size(self.current_size)
                if self._has_cutmixupaug:
                    self.learn.cut_mix_up_augment._size = _to_size(self.current_size)
            else:
                # Reset everything after progressive resizing is done
                if self.null_resize:
                    resize.size = None
                    if self._has_cutmixupaug:
                        self.learn.cut_mix_up_augment._size = None
                else:
                    resize.size = _to_size(self.current_size)
                    resize.mode = self._orig_modes[i]

        if (self.current_size == self.final_size).all() and self.remove_resize:
            self.add_resize = False
            if self.remove_cutmix:
                self.learn.cut_mix_up_augment._cutmixaugs_pipe = Pipeline([])
                self.learn.cut_mix_up_augment._docutmixaug = False

        if self.has_logger:
            self._log_size()

    def _process_pipeline(self, pipe, remove_resize=False, null_resize=None):
        'Helper method for processing augmentation pipelines'
        for p in pipe:
            if isinstance(p, _resize_augs):
                self._resize.append(p)
                if null_resize is None:
                    self.null_resize = self.null_resize and p.size is None
                else:
                    self.null_resize = null_resize
        self.remove_resize = remove_resize

# %% ../../nbs/callback.progresize.ipynb 38
try:
    import wandb

    @patch
    def _wandb_log_size(self:ProgressiveResize, next_step=True):
        size = _to_size(self.current_size)
        wandb.log({'progressive_resize_size': size[0]}, self.learn.wandb._wandb_step+int(next_step))
except:
    pass
