# fastxtend
> fastxtend (fastai extended) is a collection of tools, extensions, and addons for fastai


## Install

To install, run:
```
pip install fastxtend
```

Or to create an editable install:
```
git clone https://github.com/warner-benjamin/fastxtend.git
cd fastxtend
pip install -e ".[dev]"
```

## Requirements

fastxtend requires fastai to be installed. See http://docs.fast.ai/ for installation instructions.

To install with dependencies for vision, audio, or all tasks run one of:
```
pip install fastxtend[vision]

pip install fastxtend[audio]

pip install fastxtend[all]
```

## Usage
Like fastai, fastxtend provides safe wildcard imports using python’s `__all__`. 
```python
from fastai.vision.all import *
from fastxtend.vision.all import *
```
In general, import fastxtend after all fastai imports, as fastxtend modifies fastai. Any method modified by fastxtend is backwards compatible with the original fastai code.

## Documentation
https://fastxtend.benjaminwarner.dev
