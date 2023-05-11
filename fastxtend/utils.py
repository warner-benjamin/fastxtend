# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/utils.ipynb.

# %% ../nbs/utils.ipynb 1
# Contains code from:
# fastai - Apache License 2.0 - Copyright (c) 2023 fast.ai
# miniai - Apache License 2.0 - Copyright (c) 2023 fast.ai
# ipython - BSD 3-Clause License - Copyright (c) 2008-Present IPython Development Team; 2001-2007 Fernando Perez; 2001 Janko Hauser; 2001 Nathaniel Gray
# mish-cuda - MIT License - Copyright (c) 2019 thomasbrandon https://github.com/thomasbrandon/mish-cuda

# %% ../nbs/utils.ipynb 3
from __future__ import annotations

import torch, random, gc, sys, traceback
import numpy as np
import PIL.Image as Image

from fastcore.foundation import contextmanager

from fastai.learner import Learner
from fastai.data.core import DataLoaders
from fastai.callback.core import set_random_states, get_random_states

# %% auto 0
__all__ = ['free_gpu_memory', 'less_random', 'scale_time', 'pil_to_numpy']

# %% ../nbs/utils.ipynb 4
def clean_ipython_hist():
    # Code in this function mainly copied from IPython source
    if not 'get_ipython' in globals():
        return
    ip = get_ipython()
    user_ns = ip.user_ns
    ip.displayhook.flush()
    pc = ip.displayhook.prompt_count + 1
    for n in range(1, pc):
        user_ns.pop('_i'+repr(n),None)
    user_ns.update(dict(_i='',_ii='',_iii=''))
    hm = ip.history_manager
    hm.input_hist_parsed[:] = [''] * pc
    hm.input_hist_raw[:] = [''] * pc
    hm._i = hm._ii = hm._iii = hm._i00 = ''

# %% ../nbs/utils.ipynb 5
def clean_traceback():
    # h/t Piotr Czapla
    if hasattr(sys, 'last_traceback'):
        traceback.clear_frames(sys.last_traceback)
        delattr(sys, 'last_traceback')
    if hasattr(sys, 'last_type'):
        delattr(sys, 'last_type')
    if hasattr(sys, 'last_value'):
        delattr(sys, 'last_value')

# %% ../nbs/utils.ipynb 6
def free_gpu_memory(learn:Learner, dls:DataLoaders=None):
    "Frees GPU memory using `gc.collect` and `torch.cuda.empty_cache`"
    learn.dls, learn, dls = None, None, None
    clean_traceback()
    clean_ipython_hist()
    gc.collect()
    torch.cuda.empty_cache()

# %% ../nbs/utils.ipynb 7
@contextmanager
def less_random(
    seed:int=42, # Seed for `random`, `torch`, and `numpy`
    deterministic:bool|None=None, # Set `torch.backends.cudnn.deterministic` if not None
    benchmark:bool|None=None # Set `torch.backends.cudnn.benchmark` if not None
):
    """
    Stores and retrieves state of random number generators. Sets random seed for `random`, `torch`, and `numpy`.

    Does not set `torch.backends.cudnn.benchmark` or `torch.backends.cudnn.deterministic` by default.
    """
    states = get_random_states()

    try: torch.manual_seed(seed)
    except NameError: pass
    try: torch.cuda.manual_seed_all(seed)
    except NameError: pass
    try: np.random.seed(seed%(2**32-1))
    except NameError: pass

    random.seed(seed)
    if deterministic is not None:
        torch.backends.cudnn.deterministic = deterministic
    if benchmark is not None:
        torch.backends.cudnn.benchmark = benchmark

    try:
        yield # we are managing global variables
    finally:
        set_random_states(**states)

# %% ../nbs/utils.ipynb 9
# modified from https://github.com/thomasbrandon/mish-cuda/blob/master/test/perftest.py
def scale_time(val:float, spec:str="#0.4G"):
    "Scale fractional second `time` values and return formatted to `spec`"
    if val == 0:
        return '-'
    PREFIXES = np.array([c for c in u"yzafpnµm kMGTPEZY"])
    exp = np.int8(np.log10(np.abs(val)) // 3 * 3 * np.sign(val))
    val /= 10.**exp
    prefix = PREFIXES[exp//3 + len(PREFIXES)//2]
    return f"{val:{spec}}{prefix}s"

# %% ../nbs/utils.ipynb 10
# From https://uploadcare.com/blog/fast-import-of-pillow-images-to-numpy-opencv-arrays/
# Up to 2.5 times faster with the same functionality and a smaller number of allocations than numpy.asarray(img)
def pil_to_numpy(img:Image.Image) -> np.ndarray:
    "Fast conversion of Pillow `Image` to NumPy NDArray"
    img.load()
    # unpack data
    enc = Image._getencoder(img.mode, 'raw', img.mode)
    enc.setimage(img.im)

    # NumPy buffer for the result
    shape, typestr = Image._conv_type_shape(img)
    data = np.empty(shape, dtype=np.dtype(typestr))
    mem = data.data.cast('B', (data.data.nbytes,))

    bufsize, s, offset = 65536, 0, 0
    while not s:
        l, s, d = enc.encode(bufsize)
        mem[offset:offset + len(d)] = d
        offset += len(d)
    if s < 0:
        raise RuntimeError("encoder error %d in tobytes" % s)
    return data
