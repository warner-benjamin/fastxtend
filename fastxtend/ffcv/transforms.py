# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/ffcv.transforms.ipynb.

# %% ../../nbs/ffcv.transforms.ipynb 1
# Contains code from:
# FFCV - Apache License 2.0 - Copyright (c) 2022 FFCV Team

# %% ../../nbs/ffcv.transforms.ipynb 2
from __future__ import annotations

import math
import numpy as np
from numpy.random import rand
from typing import Callable, Optional, Tuple
from dataclasses import replace

from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State

from ffcv.transforms.cutout import Cutout
from ffcv.transforms.flip import RandomHorizontalFlip as _RandomHorizontalFlip
from ffcv.transforms.random_resized_crop import RandomResizedCrop
from ffcv.transforms.translate import RandomTranslate
from ffcv.transforms.color_jitter import RandomBrightness as _RandomBrightness
from ffcv.transforms.color_jitter import RandomContrast as _RandomContrast
from ffcv.transforms.color_jitter import RandomSaturation as _RandomSaturation
from ffcv.transforms.poisoning import Poison
from ffcv.transforms.replace_label import ReplaceLabel
from ffcv.transforms.common import Squeeze

# %% auto 0
__all__ = ['RandomHorizontalFlip', 'RandomBrightness', 'RandomContrast', 'RandomSaturation', 'RandomLighting', 'RandomHue',
           'RandomCutout', 'RandomErasing', 'RandomResizedCrop', 'RandomTranslate', 'Cutout', 'Poison', 'ReplaceLabel',
           'Squeeze']

# %% ../../nbs/ffcv.transforms.ipynb 3
_all_ = ['RandomResizedCrop', 'RandomTranslate', 'Cutout', 'Poison', 'ReplaceLabel', 'Squeeze']

# %% ../../nbs/ffcv.transforms.ipynb 9
class RandomHorizontalFlip(_RandomHorizontalFlip):
    """
    Flip the image horizontally with probability flip_prob.
    Operates on raw arrays (not tensors).

    Parameters
    ----------
    prob : float
        The probability with which to flip each image in the batch
        horizontally.
    """

    def __init__(self, prob: float = 0.5):
        super().__init__(prob)

# %% ../../nbs/ffcv.transforms.ipynb 16
class RandomBrightness(Operation):
    'Randomly adjust image brightness. Supports both TorchVision and fastai style brighness transforms.'
    def __init__(self,
        prob:float, # Probability of changing brightness
        max_lighting:float, # Maximum brightness change. Randomly choose factor on [max(0, 1-magnitude), 1+magnitude], or [0.5*(1-magnitude), 0.5*(1+magnitude)] if fastai=True.
        fastai:bool=False # If True applies the slower, fastai-style transform. Defaults to TorchVision.
    ):
        super().__init__()
        self.prob = prob
        self.magnitude = max_lighting
        self.fastai = fastai

    def generate_code(self):
        my_range = Compiler.get_iterator()
        prob = self.prob
        magnitude = self.magnitude

        if self.fastai:
            def brightness(images, dst):
                fp = np.float32
                def logit(x):
                    return -np.log(fp(1) / x - fp(1))
                def sigmoid(x):
                    return fp(1) / (fp(1) + np.exp(-x))

                apply_bright = np.random.rand(images.shape[0]) < prob
                magnitudes = logit(np.random.uniform(0.5*(1-magnitude), 0.5*(1+magnitude), images.shape[0]).astype(fp))
                for i in my_range(images.shape[0]):
                    if apply_bright[i]:
                        img = images[i] / fp(255)
                        img = logit(img)
                        img = sigmoid(img + magnitudes[i])
                        dst[i] = (img*255).astype(np.uint8)
                    else:
                        dst[i] = images[i]
                return dst
        else:
            def brightness(images, dst):
                fp = np.float32
                def blend(img1, img2, ratio): 
                    return (ratio*img1 + (1-ratio)*img2).clip(0, 255).astype(img1.dtype)

                apply_bright = np.random.rand(images.shape[0]) < prob
                magnitudes = np.random.uniform(max(0, 1-magnitude), 1+magnitude, images.shape[0]).astype(fp)
                for i in my_range(images.shape[0]):
                    if apply_bright[i]:
                        dst[i] = blend(images[i], 0, magnitudes[i])
                    else:
                        dst[i] = images[i]
                return dst

        brightness.is_parallel = True
        return brightness

    def declare_state_and_memory(self, previous_state) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), AllocationQuery(previous_state.shape, previous_state.dtype))

# %% ../../nbs/ffcv.transforms.ipynb 17
class RandomContrast(Operation):
    'Randomly adjust image contrast. Supports both TorchVision and fastai style contrast transforms.'
    def __init__(self,
        prob:float, # Probability of changing contrast
        max_lighting:float, # Maximum contrast change. Randomly choose factor on [max(0, 1-magnitude), 1+magnitude], or [1-max_lighting, 1/(1-max_lighting)] in log space if fastai=True.
        fastai:bool=False # If True applies the slower, fastai-style transform. Defaults to TorchVision.
    ):
        super().__init__()
        self.prob = prob
        self.magnitude = max_lighting
        self.fastai = fastai

    def generate_code(self):
        my_range = Compiler.get_iterator()
        prob = self.prob
        magnitude = self.magnitude

        if self.fastai:
            def contrast(images, dst):
                fp = np.float32
                def logit(x):
                    return -np.log(fp(1) / x - fp(1))
                def sigmoid(x):
                    return fp(1) / (fp(1) + np.exp(-x))

                apply_contrast = np.random.rand(images.shape[0]) < prob
                magnitudes = np.exp(np.random.uniform(np.log(1-magnitude), -np.log(1-magnitude), images.shape[0]).astype(fp))
                for i in my_range(images.shape[0]):
                    if apply_contrast[i]:
                        img = images[i] / fp(255)
                        img = logit(img)
                        img = sigmoid(img * magnitudes[i])
                        dst[i] = (img*255).astype(np.uint8)
                    else:
                        dst[i] = images[i]
                return dst
        else:
            def contrast(images, dst):
                fp = np.float32
                def blend(img1, img2, ratio): 
                    return (ratio*img1 + (1-ratio)*img2).clip(0, 255).astype(img1.dtype)

                apply_contrast = np.random.rand(images.shape[0]) < prob
                magnitudes = np.random.uniform(max(0, 1-magnitude), 1+magnitude, images.shape[0]).astype(fp)
                for i in my_range(images.shape[0]):
                    if apply_contrast[i]:
                        l_img = fp(0.2989)*images[i,:,:,0] + fp(0.587)*images[i,:,:,1] + fp(0.114)*images[i,:,:,2]
                        dst[i] = blend(images[i], l_img.mean(), magnitudes[i])
                    else:
                        dst[i] = images[i]
                return dst

        contrast.is_parallel = True
        return contrast

    def declare_state_and_memory(self, previous_state) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), AllocationQuery(previous_state.shape, previous_state.dtype))

# %% ../../nbs/ffcv.transforms.ipynb 18
class RandomSaturation(Operation):
    'Randomly adjust image saturation. Supports both TorchVision and fastai style contrast transforms.'
    def __init__(self,
        prob:float, # Probability of changing saturation
        max_lighting:float, # Maximum saturation change. Randomly choose factor on [max(0, 1-magnitude), 1+magnitude], or [1-max_lighting, 1/(1-max_lighting)] in log space if fastai=True.
        fastai:bool=False # If True applies the slower, fastai-style transform. Defaults to TorchVision.
    ):
        super().__init__()
        self.prob = prob
        self.magnitude = max_lighting
        self.fastai = fastai

    def generate_code(self):
        my_range = Compiler.get_iterator()
        prob = self.prob
        magnitude = self.magnitude

        if self.fastai:
            def saturation(images, dst):
                fp = np.float32
                def logit(x):
                    return -np.log(fp(1) / x - fp(1))
                def sigmoid(x):
                    return fp(1) / (fp(1) + np.exp(-x))
                def grayscale(x):
                    return fp(0.2989) * x[:,:,0] + fp(0.587) * x[:,:,1] + fp(0.114) * x[:,:,2]

                apply_saturation = np.random.rand(images.shape[0]) < prob
                magnitudes = np.exp(np.random.uniform(np.log(1-magnitude), -np.log(1-magnitude), images.shape[0]).astype(fp))
                for i in my_range(images.shape[0]):
                    if apply_saturation[i]:
                        img = images[i] / fp(255)

                        l_img = grayscale(img) * (1-magnitudes[i])
                        gray = np.empty_like(img)
                        for j in range(3):
                            gray[:,:,j] = l_img

                        img = logit(img)
                        img = img * magnitudes[i]
                        img = sigmoid(img + gray)
                        dst[i] = (img*255).astype(np.uint8)
                    else:
                        dst[i] = images[i]
                return dst
        else:
            def saturation(images, dst):
                fp = np.float32
                def blend(img1, img2, ratio): 
                    return (ratio*img1 + (1-ratio)*img2).clip(0, 255).astype(img1.dtype)

                apply_saturation = np.random.rand(images.shape[0]) < prob
                magnitudes = np.random.uniform(max(0, 1-magnitude), 1+magnitude, images.shape[0]).astype(fp)
                for i in my_range(images.shape[0]):
                    if apply_saturation[i]:
                        l_img = fp(0.2989)*images[i,:,:,0] + fp(0.587)*images[i,:,:,1] + fp(0.114)*images[i,:,:,2]
                        gray = np.empty_like(l_img)
                        for j in range(3):
                            gray[:,:,j] = l_img
                        dst[i] = blend(images[i], gray, magnitudes[i])
                    else:
                        dst[i] = images[i]
                return dst

        saturation.is_parallel = True
        return saturation

    def declare_state_and_memory(self, previous_state) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), AllocationQuery(previous_state.shape, previous_state.dtype))

# %% ../../nbs/ffcv.transforms.ipynb 19
class RandomLighting(Operation):
    '''
    Randomly adjust image brightness, contrast, and saturation. 
    Combines all three into single transform for speed.
    Supports both TorchVision and fastai style contrast transforms. 
    '''
    def __init__(self,
        prob:float|None, # Probability of changing brightness, contrast, and saturation. Set to None for individual probability.
        max_lighting:float|None, # Maximum lighting change. Set to None for individual probability. See max_brightness, max_contrast, and max_saturation for details.
        max_brightness:float|None=None, # Maximum brightness change. Randomly choose factor on [max(0, 1-magnitude), 1+magnitude], or [0.5*(1-magnitude), 0.5*(1+magnitude)] if fastai=True.
        max_contrast:float|None=None, # Maximum contrast change. Randomly choose factor on [max(0, 1-magnitude), 1+magnitude], or [1-max_lighting, 1/(1-max_lighting)] in log space if fastai=True.
        max_saturation:float|None=None, # Maximum saturation change. Randomly choose factor on [max(0, 1-magnitude), 1+magnitude], or [1-max_lighting, 1/(1-max_lighting)] in log space if fastai=True.
        prob_brightness:float|None=None, # Individual probability of changing brightness. Set to prob=None to use.
        prob_contrast:float|None=None, # Individual probability of changing contrast. Set to prob=None to use.
        prob_saturation:float|None=None, # Individual probability of changing saturation. Set to prob=None to use.
        fastai:bool=False # If True applies the slower, fastai-style transform. Defaults to TorchVision.
    ):
        super().__init__()
        self.prob = prob
        self.fastai = fastai
        self.max_lighting    = max_lighting
        self.max_brightness  = max_brightness
        self.max_contrast    = max_contrast
        self.max_saturation  = max_saturation
        self.prob_brightness = prob_brightness
        self.prob_contrast   = prob_contrast
        self.prob_saturation = prob_saturation

    def generate_code(self):
        my_range = Compiler.get_iterator()
        if self.prob is not None:
            prob_brightness, prob_contrast, prob_saturation = self.prob, self.prob, self.prob
        else:
            prob_brightness, prob_contrast, prob_saturation = self.prob_brightness, self.prob_contrast, self.prob_saturation
        if self.max_lighting is not None:
            max_brightness, max_contrast, max_saturation = self.max_lighting, self.max_lighting, self.max_lighting
        else:
            max_brightness, max_contrast, max_saturation = self.max_brightness, self.max_contrast, self.max_saturation

        if self.fastai:
            def lighting(images, dst):
                fp = np.float32
                assert images.shape[-1] == 3
                def logit(x):
                    return -np.log(fp(1) / x - fp(1))
                def sigmoid(x):
                    return fp(1) / (fp(1) + np.exp(-x))
                def grayscale(x):
                    return fp(0.2989) * x[:,:,0] + fp(0.587) * x[:,:,1] + fp(0.114) * x[:,:,2]
                def probs(max, shape, prob):
                    return np.random.rand(shape) < prob if max > 0 else np.zeros(shape)==1

                bs = images.shape[0]
                apply_brightness = probs(max_brightness, bs, prob_brightness)
                apply_contrast   = probs(max_contrast, bs, prob_contrast)
                apply_saturation = probs(max_saturation, bs, prob_saturation)

                brightness = logit(np.random.uniform(0.5*(1-max_brightness), 0.5*(1+max_brightness), bs).astype(fp))
                contrast   = np.exp(np.random.uniform(np.log(1-max_contrast), -np.log(1-max_contrast), bs).astype(fp))
                saturation = np.exp(np.random.uniform(np.log(1-max_saturation), -np.log(1-max_saturation), bs).astype(fp))
                for i in my_range(bs):
                    if apply_brightness[i] or apply_contrast[i] or apply_saturation[i]:
                        img = images[i] / fp(255)

                        if apply_saturation[i]:
                            l_img = grayscale(img)
                        else:
                            l_img = np.empty_like(img[:,:,0])
                        
                        img = logit(img)

                        if apply_brightness[i]:
                            img = img + brightness[i]

                        if apply_contrast[i]:
                            img = img * contrast[i]

                        if apply_saturation[i]:
                            l_img = l_img * (fp(1)-saturation[i])
                            gray = np.empty_like(img)
                            for j in range(3):
                                gray[:,:,j] = l_img
                            img = img * saturation[i]
                            img = img + gray

                        img = sigmoid(img)
                        dst[i] = (img*255).astype(np.uint8)
                    else:
                        dst[i] = images[i]
                return dst
        else:
            def lighting(images, dst):
                fp = np.float32
                def blend(img1, img2, ratio): 
                    return (ratio*img1 + (1-ratio)*img2).clip(0, 255).astype(img1.dtype)
                def probs(max, shape, prob):
                    return np.random.rand(shape) < prob if max > 0 else np.zeros(shape)==1

                bs = images.shape[0]
                apply_brightness = probs(max_brightness, bs, prob_brightness)
                apply_contrast   = probs(max_contrast, bs, prob_contrast)
                apply_saturation = probs(max_saturation, bs, prob_saturation)

                brightness = np.random.uniform(max(0, 1-max_brightness), 1+max_brightness, images.shape[0]).astype(fp)
                contrast = np.random.uniform(max(0, 1-max_contrast), 1+max_contrast, images.shape[0]).astype(fp)
                saturation = np.random.uniform(max(0, 1-max_saturation), 1+max_saturation, images.shape[0]).astype(fp)
                for i in my_range(images.shape[0]):
                    dst[i] = images[i]
                    if apply_brightness[i] or apply_contrast[i] or apply_saturation[i]:
                        if apply_brightness[i]:
                            dst[i] = blend(dst[i], 0, brightness[i])

                        if apply_contrast[i] or apply_saturation[i]:
                            l_img = fp(0.2989)*dst[i,:,:,0] + fp(0.587)*dst[i,:,:,1] + fp(0.114)*dst[i,:,:,2]
                        else:
                            l_img = np.empty_like(dst[i,:,:,0], dtype=np.float32)

                        if apply_contrast[i]:
                            dst[i] = blend(dst[i], l_img.mean(), contrast[i])

                        if apply_saturation[i]:
                            gray = np.empty_like(dst[i], dtype=np.float32)
                            for j in range(3):
                                gray[:,:,j] = l_img
                            dst[i] = blend(dst[i], gray, saturation[i])
                return dst

        lighting.is_parallel = True
        return lighting

    def declare_state_and_memory(self, previous_state) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), AllocationQuery(previous_state.shape, previous_state.dtype))

# %% ../../nbs/ffcv.transforms.ipynb 20
# RandomHue adapted from pending FFCV PR: https://github.com/libffcv/ffcv/pull/226

# Code for Hue adapted from:
# https://sanje2v.wordpress.com/2021/01/11/accelerating-data-transforms/
# https://stackoverflow.com/questions/8507885

class RandomHue(Operation):
    'Randomly adjust image Hue. Supports both TorchVision and fastai style contrast transforms.'
    def __init__(self,
        prob:float, # Probability of changing hue
        max_hue:float, # Maximum hue change. Randomly choose factor on [-magnitude, magnitude] clipped to [-0.5, 0.5] or [1-max_hue, 1/(1-max_hue)] in log space if fastai=True.
        fastai:bool=False # If True applies the slower, fastai-style transform. Defaults to TorchVision
    ):
        super().__init__()
        self.prob = prob
        self.magnitude = max_hue if fastai else np.clip(max_hue, -0.5, 0.5)
        self.fastai = fastai

    def generate_code(self):
        my_range = Compiler.get_iterator()
        prob = self.prob
        magnitude = self.magnitude
        fastai = self.fastai

        def hue(images, dst):
            fp = np.float32
            sqrt3 = np.sqrt(fp(1/3))
            apply_hue = np.random.rand(images.shape[0]) < prob
            if fastai:
                magnitudes = np.random.uniform(np.log(1-magnitude), -np.log(1-magnitude), images.shape[0]).astype(fp)
            else:
                magnitudes = np.random.uniform(-magnitude, magnitude, images.shape[0]).astype(fp)
            for i in my_range(images.shape[0]):
                if apply_hue[i] and magnitudes[i]!=0:
                    img = images[i] / fp(255)
                    hue_factor_radians = magnitudes[i] * fp(2) * fp(np.pi)
                    cosA = np.cos(hue_factor_radians)
                    sinA = np.sin(hue_factor_radians)
                    hue_rotation_matrix =\
                        [[cosA + (fp(1) - cosA) / fp(3), fp(1/3) * (fp(1) - cosA) - sqrt3 * sinA, fp(1/3) * (fp(1) - cosA) + sqrt3 * sinA],
                        [fp(1/3) * (fp(1) - cosA) + sqrt3 * sinA, cosA + fp(1/3)*(fp(1) - cosA), fp(1/3) * (fp(1) - cosA) - sqrt3 * sinA],
                        [fp(1/3) * (fp(1) - cosA) - sqrt3 * sinA, fp(1/3) * (fp(1) - cosA) + sqrt3 * sinA, cosA + fp(1/3) * (fp(1) - cosA)]]
                    hue_rotation_matrix = np.array(hue_rotation_matrix, dtype=img.dtype)

                    for row in range(img.shape[0]):
                        for col in range(img.shape[1]):
                            r, g, b = img[row, col, :]
                            img[row, col, 0] = r * hue_rotation_matrix[0, 0] + g * hue_rotation_matrix[0, 1] + b * hue_rotation_matrix[0, 2]
                            img[row, col, 1] = r * hue_rotation_matrix[1, 0] + g * hue_rotation_matrix[1, 1] + b * hue_rotation_matrix[1, 2]
                            img[row, col, 2] = r * hue_rotation_matrix[2, 0] + g * hue_rotation_matrix[2, 1] + b * hue_rotation_matrix[2, 2]
                    dst[i] = np.clip(img * 255, 0, 255).astype(np.uint8)
                else:
                    dst[i] = images[i]
            return dst

        hue.is_parallel = True
        return hue

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), AllocationQuery(previous_state.shape, previous_state.dtype))

# %% ../../nbs/ffcv.transforms.ipynb 22
class RandomCutout(Cutout):
    """Random cutout data augmentation (https://arxiv.org/abs/1708.04552).

    Parameters
    ----------
    prob : float
        Probability of applying on each image.
    crop_size : int
        Size of the random square to cut out.
    fill : Tuple[int, int, int], optional
        An RGB color ((0, 0, 0) by default) to fill the cutout square with.
        Useful for when a normalization layer follows cutout, in which case
        you can set the fill such that the square is zero post-normalization.
    """
    def __init__(self, prob: float, crop_size: int, fill: Tuple[int, int, int] = (0, 0, 0)):
        super().__init__(crop_size, fill)
        self.prob = np.clip(prob, 0., 1.)

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        crop_size = self.crop_size
        fill = self.fill
        prob = self.prob

        def cutout_square(images, *_):
            should_cutout = rand(images.shape[0]) < prob
            for i in my_range(images.shape[0]):
                if should_cutout[i]:
                    # Generate random origin
                    coord = (
                        np.random.randint(images.shape[1] - crop_size + 1),
                        np.random.randint(images.shape[2] - crop_size + 1),
                    )
                    # Black out image in-place
                    images[i, coord[0]:coord[0] + crop_size, coord[1]:coord[1] + crop_size] = fill
            return images

        cutout_square.is_parallel = True
        return cutout_square

# %% ../../nbs/ffcv.transforms.ipynb 23
# Implementation inspired by fastai https://docs.fast.ai/vision.augment.html#randomerasing
# fastai - Apache License 2.0 - Copyright (c) 2023 fast.ai
class RandomErasing(Operation):
    """Random erasing data augmentation (https://arxiv.org/abs/1708.04896).

    Parameters
    ----------
    prob : float
        Probability of applying on each image.
    min_area : float
        Minimum erased area as percentage of image size.
    max_area : float
        Maximum erased area as percentage of image size.
    min_aspect : float
        Minimum aspect ratio of erased area.
    max_count : int
        Maximum number of erased blocks per image. Erased Area is scaled by max_count.
    fill_mean : Tuple[int, int, int], optional
        The RGB color mean (ImageNet's (124, 116, 103) by default) to randomly fill the
        erased area with. Should be the mean of dataset or pretrained dataset.
    fill_std : Tuple[int, int, int], optional
        The RGB color standard deviation (ImageNet's (58, 57, 57) by default) to randomly
        fill the erased area with. Should be the st. dev of dataset or pretrained dataset.
    fast_fill : bool
        Default of True is ~2X faster by generating noise once per batch and randomly
        selecting slices of the noise instead of generating unique noise per each image.
    """
    def __init__(self, prob: float, min_area: float = 0.02, max_area: float = 0.3,
                 min_aspect: float = 0.3, max_count: int = 1,
                 fill_mean: Tuple[int, int, int] = (124, 116, 103),
                 fill_std: Tuple[int, int, int] = (58, 57, 57),
                 fast_fill : bool = True):
        super().__init__()
        self.prob = np.clip(prob, 0., 1.)
        self.min_area = np.clip(min_area, 0., 1.)
        self.max_area = np.clip(max_area, 0., 1.)
        self.log_ratio = (math.log(np.clip(min_aspect, 0., 1.)), math.log(1/np.clip(min_aspect, 0., 1.)))
        self.max_count = max_count
        self.fill_mean = np.array(fill_mean)
        self.fill_std = np.array(fill_std)
        self.fast_fill = fast_fill

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        prob = self.prob
        min_area = self.min_area
        max_area = self.max_area
        log_ratio = self.log_ratio
        max_count = self.max_count
        fill_mean = self.fill_mean
        fill_std = self.fill_std
        fast_fill = self.fast_fill

        def random_erase(images, *_):
            if fast_fill:
                noise = fill_mean + (fill_std * np.random.randn(images.shape[1], images.shape[2], images.shape[3])).astype(images.dtype)

            should_cutout = rand(images.shape[0]) < prob
            for i in my_range(images.shape[0]):
                if should_cutout[i]:
                    count = np.random.randint(1, max_count) if max_count > 1 else 1
                    for j in range(count):
                        # Randomly select bounds
                        area = np.random.uniform(min_area, max_area, 1) * images.shape[1] * images.shape[2] / count
                        aspect = np.exp(np.random.uniform(log_ratio[0], log_ratio[1], 1))
                        bound = (
                            int(round(np.sqrt(area * aspect).item())),
                            int(round(np.sqrt(area / aspect).item())),
                        )
                        # Select random erased area
                        coord = (
                            np.random.randint(0, max(1, images.shape[1] - bound[0])),
                            np.random.randint(0, max(1, images.shape[2] - bound[1])),
                        )
                        # Fill image with random noise in-place
                        if fast_fill:
                            images[i, coord[0]:coord[0] + bound[0], coord[1]:coord[1] + bound[1]] =\
                                noise[coord[0]:coord[0] + bound[0], coord[1]:coord[1] + bound[1]]
                        else:
                            noise = fill_mean + (fill_std * np.random.randn(bound[0], bound[1], images.shape[3])).astype(images.dtype)
                            images[i, coord[0]:coord[0] + bound[0], coord[1]:coord[1] + bound[1]] = noise
            return images

        random_erase.is_parallel = True
        return random_erase

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return replace(previous_state, jit_mode=True), None
