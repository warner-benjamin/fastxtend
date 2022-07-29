# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/callback.cutmixup.ipynb (unless otherwise specified).


from __future__ import annotations


__all__ = ['MixUp', 'CutMix', 'CutMixUp', 'CutMixUpAugment']

# Cell
#nbdev_comment from __future__ import annotations

from torch.distributions import Bernoulli, Categorical
from torch.distributions.beta import Beta

from fastcore.transform import Pipeline, Transform

from fastai.data.transforms import IntToFloatTensor, Normalize
from fastai.callback.mixup import reduce_loss
from fastai.layers import NoneReduce
from fastai.vision.augment import AffineCoordTfm, RandomResizedCropGPU

from ..multiloss import MixHandlerX
from ..imports import *

# Cell
class MixUp(MixHandlerX):
    "Implementation of https://arxiv.org/abs/1710.09412"
    def __init__(self,
        alpha:float=.4, # Alpha & beta parametrization for `Beta` distribution
        interp_label:bool|None=None # Blend or stack labels. Defaults to loss' `y_int` if None
    ):
        super().__init__(alpha, interp_label)

    def before_batch(self):
        "Blend inputs and labels with another random item in the batch"
        shuffle = torch.randperm(self.y.size(0)).to(self.x.device)
        xb1,self.yb1 = tuple(L(self.xb).itemgot(shuffle)),tuple(L(self.yb).itemgot(shuffle))
        self.lam = self._mixup(self.y.size(0))
        nx_dims = len(self.x.size())
        self.learn.xb = tuple(L(xb1,self.xb).map_zip(torch.lerp,weight=unsqueeze(self.lam, n=nx_dims-1)))

        if not self.stack_y:
            ny_dims = len(self.y.size())
            self.learn.yb = tuple(L(self.yb1,self.yb).map_zip(torch.lerp,weight=unsqueeze(self.lam, n=ny_dims-1)))

    def _mixup(self, bs):
        lam = self.distrib.sample((bs,)).squeeze().to(self.x.device)
        lam = torch.stack([lam, 1-lam], 1)
        return lam.max(1)[0]

# Cell
class CutMix(MixHandlerX):
    "Implementation of https://arxiv.org/abs/1905.04899"
    def __init__(self,
        alpha:float=1., # Alpha & beta parametrization for `Beta` distribution
        uniform:bool=True, # Uniform patches across batch. True matches fastai CutMix
        interp_label:bool|None=None # Blend or stack labels. Defaults to loss' `y_int` if None
    ):
        super().__init__(alpha, interp_label)
        store_attr(but='alpha,interp_label')

    def before_batch(self):
        "Add patches and blend labels from another random item in batch"
        bs, _, H, W = self.x.size()
        shuffle = torch.randperm(bs).to(self.x.device)
        xb1,self.yb1 = self.x[shuffle], (self.y[shuffle],)

        if self.uniform:
            xb, self.lam = self._uniform_cutmix(self.x, xb1, H, W)
        else:
            xb, self.lam = self._multi_cutmix(self.x, xb1, H, W, bs)
        self.learn.xb = (xb,)

        if not self.stack_y:
            ny_dims = len(self.y.size())
            self.learn.yb = tuple(L(self.yb1,self.yb).map_zip(torch.lerp,weight=unsqueeze(self.lam, n=ny_dims-1)))

    def _uniform_cutmix(self, xb, xb1, H, W):
        "Add uniform patches and blend labels from another random item in batch"
        self.lam = self.distrib.sample((1,)).to(self.x.device)
        x1, y1, x2, y2 = self.rand_bbox(W, H, self.lam)
        xb[..., y1:y2, x1:x2] = xb1[..., y1:y2, x1:x2]
        lam = (1 - ((x2-x1)*(y2-y1))/float(W*H))
        return xb, lam

    def _multi_cutmix(self, xb, xb1, H, W, bs):
        "Add random patches and blend labels from another random item in batch"
        lam = self.distrib.sample((bs,)).to(self.x.device)
        for i in range(bs):
            if 1 > lam[i] > 0:
                x1, y1, x2, y2 = self.rand_bbox(W, H, lam[i])
                xb[i, ..., y1:y2, x1:x2] = xb1[i, ..., y1:y2, x1:x2]
                lam[i] = (1 - ((x2-x1)*(y2-y1))/float(W*H))
        return xb, lam

    def rand_bbox(self,
        W:int, # Input image width
        H:int, # Input image height
        lam:Tensor # Lambda sample from Beta distribution
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]: # Top-left and bottom-right coordinates
        "Return random sub coordinates"
        cut_rat = torch.sqrt(1. - lam).to(self.x.device)
        cut_w = torch.round(W * cut_rat).type(torch.long).to(self.x.device)
        cut_h = torch.round(H * cut_rat).type(torch.long).to(self.x.device)
        # uniform
        cx = torch.randint(0, W, (1,)).to(self.x.device)
        cy = torch.randint(0, H, (1,)).to(self.x.device)
        x1 = torch.clamp(cx - torch.div(cut_w, 2, rounding_mode='floor'), 0, W)
        y1 = torch.clamp(cy - torch.div(cut_h, 2, rounding_mode='floor'), 0, H)
        x2 = torch.clamp(cx + torch.div(cut_w, 2, rounding_mode='floor'), 0, W)
        y2 = torch.clamp(cy + torch.div(cut_h, 2, rounding_mode='floor'), 0, H)
        return x1, y1, x2, y2

# Cell
class CutMixUp(MixUp, CutMix):
    "Combo implementation of https://arxiv.org/abs/1710.09412 and https://arxiv.org/abs/1905.04899"
    run_valid = False
    def __init__(self,
        mix_alpha:float=.4, # MixUp alpha & beta parametrization for `Beta` distribution
        cut_alpha:float=1., # CutMix alpha & beta parametrization for `Beta` distribution
        mixup_ratio:Number=1, # Ratio to apply `MixUp` relative to `CutMix`
        cutmix_ratio:Number=1, # Ratio to apply `CutMix` relative to `MixUp`
        cutmix_uniform:bool=False, # Uniform patches across batch. True matches fastai CutMix
        same_batch:bool=True, # Apply MixUp and CutMix on the same batch
        interp_label:bool|None=None # Blend or stack labels. Defaults to loss' `y_int` if None
    ):
        store_attr()
        if same_batch:
            total = mixup_ratio + cutmix_ratio
            self.categorical = Categorical(tensor([mixup_ratio/total, cutmix_ratio/total]))
        MixUp.__init__(self, mix_alpha, interp_label)
        CutMix.__init__(self, cut_alpha, cutmix_uniform, interp_label)
        self.mix_distrib = Beta(tensor(mix_alpha), tensor(mix_alpha))
        self.cut_distrib = Beta(tensor(cut_alpha), tensor(cut_alpha))
        self.ratio = mixup_ratio / (cutmix_ratio + mixup_ratio)

    def before_batch(self):
        "Apply MixUp or CutMix"
        if self.same_batch:
            xb, self.yb1 = self.x, self.y
            bs, _, _, _ = xb.size()
            self.lam = torch.zeros(bs, device=xb.device)
            aug_type = self.categorical.sample((bs,))
            shuffle = torch.randperm(xb.shape[0]).to(xb.device)
            xb1, self.yb1 = xb[shuffle], self.yb1[shuffle]

            # Apply MixUp
            self.distrib = self.mix_distrib
            self.lam[aug_type==0] = MixUp._mixup(self, xb[aug_type==0].shape[0])
            xb[aug_type==0] = torch.lerp(xb1[aug_type==0], xb[aug_type==0], weight=unsqueeze(self.lam[aug_type==0], n=3))

            # Apply CutMix
            bs, _, H, W = xb[aug_type==1].size()
            self.distrib = self.cut_distrib
            if self.cutmix_uniform:
                xb[aug_type==1], self.lam[aug_type==1] = CutMix._uniform_cutmix(self, xb[aug_type==1], xb1[aug_type==1], H, W)
            else:
                xb[aug_type==1], self.lam[aug_type==1] = CutMix._multi_cutmix(self, xb[aug_type==1], xb1[aug_type==1], H, W, bs)

            self.learn.xb = (xb,)
            if not self.stack_y:
                ny_dims = len(self.yb1.size())
                self.learn.yb = tuple(L(self.yb1,self.yb).map_zip(torch.lerp,weight=unsqueeze(self.lam, n=ny_dims-1)))
            else:
                self.yb1 = (self.yb1,)

        elif torch.rand(1) <= self.ratio: #mixup
            self.distrib = self.mix_distrib
            MixUp.before_batch(self)
        else:
            self.distrib = self.cut_distrib
            CutMix.before_batch(self)

# Cell
class CutMixUpAugment(MixUp, CutMix):
    """
    Combo implementation of https://arxiv.org/abs/1710.09412 and https://arxiv.org/abs/1905.04899 plus Augmentation.

    Pulls augmentations from `Dataloaders.train.after_batch`. These augmentations are not applied when performing `MixUp` & `CutMix`, the frequency is controlled by `augment_ratio`.

    Use `augment_finetune` to only apply dataloader augmentations at the end of training.

    `cutmixup_augs` are an optional separate set of augmentations to apply with `MixUp` and `CutMix`. Usually less intensive then the dataloader augmentations.
    """
    run_valid = False
    def __init__(self,
        mix_alpha:float=.4, # MixUp alpha & beta parametrization for `Beta` distribution
        cut_alpha:float=1., # CutMix alpha & beta parametrization for `Beta` distribution
        mixup_ratio:Number=1, # Ratio to apply `MixUp` relative to `CutMix` & augmentations
        cutmix_ratio:Number=1, # Ratio to apply `CutMix` relative to `MixUp` & augmentations
        augment_ratio:Number=1, # Ratio to apply augmentations relative to `MixUp` & `CutMix`
        augment_finetune:Number|None=None, # Number of epochs or pct of training to only apply augmentations
        cutmix_uniform:bool=False, # Uniform patches across batch. True matches fastai CutMix
        cutmixup_augs:listified[Transform|Callable[...,Transform]]|None=None, # Augmentations to apply before `MixUp` & `CutMix`. Should not have `Normalize`
        same_batch:bool=True, # Apply MixUp, CutMix, and Augment on the same batch
        interp_label:bool|None=None, # Blend or stack labels. Defaults to loss' `y_int` if None
    ):
        store_attr()
        if same_batch:
            total = mixup_ratio + cutmix_ratio + augment_ratio
            self.categorical = Categorical(tensor([mixup_ratio/total, cutmix_ratio/total, augment_ratio/total]))
        MixUp.__init__(self, mix_alpha, interp_label)
        CutMix.__init__(self, cut_alpha, cutmix_uniform, interp_label)
        self.mix_distrib = Beta(tensor(mix_alpha), tensor(mix_alpha))
        self.cut_distrib = Beta(tensor(cut_alpha), tensor(cut_alpha))
        self.aug_cutmix_ratio = augment_ratio / (augment_ratio + cutmix_ratio + mixup_ratio)
        if self.aug_cutmix_ratio == 1: self.cut_mix_ratio = 0
        else: self.cut_mix_ratio = mixup_ratio / (cutmix_ratio + mixup_ratio)
        self._docutmixaug = cutmixup_augs is not None

    def before_fit(self):
        "Remove training augmentations from dataloader & setup augmentation pipelines"
        super().before_fit()
        if self.augment_finetune is None: self.augment_finetune = (self.learn.n_epoch + 1)/self.learn.n_epoch
        elif self.augment_finetune >= 1: self.augment_finetune = self.augment_finetune/self.learn.n_epoch
        else: self.augment_finetune = self.augment_finetune

        self._inttofloat_pipe = Pipeline([])
        self._norm_pipe = Pipeline([])
        if self._docutmixaug:
            self._cutmixaugs_pipe = Pipeline(self.cutmixup_augs)

        # first copy transforms
        self._orig_pipe = self.dls.train.after_batch
        self._orig_pipe.split_idx = 0 # need to manually set split_idx for training augmentations to run

        # Loop through existing transforms looking for IntToFloatTensor, Normalize
        self._size, mode, augs = None, None, []
        for aug in self.dls.train.after_batch.fs:
            if isinstance(aug, IntToFloatTensor):
                self._inttofloat_pipe = Pipeline([aug])
            else:
                if isinstance(aug, Normalize):
                    self._norm_pipe = Pipeline([aug])
                elif isinstance(aug, (AffineCoordTfm, RandomResizedCropGPU)) and aug.size is not None:
                    self._size = aug.size
                    mode = aug.mode
                if self.same_batch:
                    augs.append(aug)

        # One Batch requires IntToFloatTensor before self._aug_pipe is called
        if self.same_batch: self._aug_pipe = Pipeline(augs)

        # If there is a resize in Augmentations and no `cutmixup_augs`, need to replicate it for MixUp/CutMix
        if not self._docutmixaug and self._size is not None:
            self._docutmixaug = True
            self._cutmixaugs_pipe = Pipeline([AffineCoordTfm(size=self._size, mode=mode)])

        # set existing transforms to an empty Pipeline
        self.dls.train.after_batch = Pipeline([])

    def before_batch(self):
        "Apply MixUp, CutMix, optional MixUp/CutMix augmentations, and/or augmentations"
        if self.same_batch and self.augment_finetune >= self.learn.pct_train:
            self._doaugs = False
            xb, self.yb1 = self.x, self.y
            bs, C, H, W = xb.size()
            self.lam = torch.zeros(bs, device=xb.device)
            aug_type = self.categorical.sample((bs,))
            shuffle = torch.randperm(xb[aug_type<2].shape[0]).to(xb.device)
            self.yb1[aug_type<2] = self.yb1[aug_type<2][shuffle]

            # Apply IntToFloat to all samples
            xb = self._inttofloat_pipe(xb)

            # New Tensor for possibly resized
            xb2 = torch.zeros([bs, C, self._size[0], self._size[1]], dtype=xb.dtype, device=xb.device) if self._size is not None else torch.zeros_like(xb)

            # Apply MixUp/CutMix Augmentations to MixUp and CutMix samples
            if self._docutmixaug:
                xb2[aug_type<2] = self._cutmixaugs_pipe(xb[aug_type<2])
            else:
                xb2[aug_type<2] = xb[aug_type<2]

            # Original Augmentations
            xb2[aug_type==2] = self._aug_pipe(xb[aug_type==2])

            # Possibly Resized xb and shuffled xb1
            xb = xb2
            xb1 = xb[aug_type<2][shuffle]

            # Apply MixUp
            self.distrib = self.mix_distrib
            self.lam[aug_type==0] = MixUp._mixup(self, xb[aug_type==0].shape[0])
            xb[aug_type==0] = torch.lerp(xb1[aug_type[aug_type<2]==0], xb[aug_type==0], weight=unsqueeze(self.lam[aug_type==0], n=3))

            # Apply CutMix
            bs, _, H, W = xb[aug_type==1].size()
            self.distrib = self.cut_distrib
            if self.cutmix_uniform:
                xb[aug_type==1], self.lam[aug_type==1] = CutMix._uniform_cutmix(self, xb[aug_type==1], xb1[aug_type[aug_type<2]==1], H, W)
            else:
                xb[aug_type==1], self.lam[aug_type==1] = CutMix._multi_cutmix(self, xb[aug_type==1], xb1[aug_type[aug_type<2]==1], H, W, bs)

            # Normalize MixUp and CutMix
            xb[aug_type<2] = self._norm_pipe(xb[aug_type<2])

            self.learn.xb = (xb,)
            if not self.stack_y:
                ny_dims = len(self.yb1.size())
                self.learn.yb = tuple(L(self.yb1,self.yb).map_zip(torch.lerp,weight=unsqueeze(self.lam, n=ny_dims-1)))
            else:
                self.yb1 = (self.yb1,)

        elif self.augment_finetune >= self.learn.pct_train and torch.rand(1) >= self.aug_cutmix_ratio: # augs or mixup/cutmix
            self._doaugs = False

            # Apply IntToFloat to MixUp/CutMix and MixUp/CutMix Augmentations
            self.learn.xb = self._inttofloat_pipe(self.xb)
            if self._docutmixaug:
                self.learn.xb = self._cutmixaugs_pipe(self.xb)

            # Perform MixUp or CutMix
            if self.cut_mix_ratio > 0 and torch.rand(1) <= self.cut_mix_ratio:
                self.distrib = self.mix_distrib
                MixUp.before_batch(self)
            else:
                self.distrib = self.cut_distrib
                CutMix.before_batch(self)

            # Normalize MixUp/CutMix
            self.learn.xb = self._norm_pipe(self.xb) # now normalize
        else:
            # Original Augmentations
            self._doaugs = True
            self.learn.xb = self._orig_pipe(self.xb)

    def after_fit(self):
        "Reset the train dataloader augmentations"
        self.dls.train.after_batch = self._orig_pipe

    def after_cancel_fit(self):
        "Reset the train dataloader augmentations and loss function"
        self.after_fit()
        MixUp.after_cancel_fit(self)

    def solo_lf(self, pred, *yb):
        "`norm_lf` applies the original loss function on both outputs based on `self.lam` if applicable"
        if not self.training or self._doaugs:
            return self.old_lf(pred, *yb)
        with NoneReduce(self.old_lf) as lf:
            loss = torch.lerp(lf(pred,*self.yb1), lf(pred,*yb), self.lam)
        return reduce_loss(loss, getattr(self.old_lf, 'reduction', 'mean'))

    def multi_lf(self, pred, *yb):
        "`norm_lf` applies the original loss function on both outputs based on `self.lam` if applicable"
        if not self.training or self._doaugs:
            return self.learn.loss_func_mixup(pred, *yb)
        else:
            return self.learn.loss_func_mixup.forward_mixup(pred, *self.yb1, *yb, self.lam)