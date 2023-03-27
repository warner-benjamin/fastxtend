conda create -n fastxtend python=3.10 pkg-config libjpeg-turbo opencv tqdm terminaltables \
psutil numpy=1.23.5 numba librosa=0.9.2 fastai nbdev timm pytorch torchvision torchaudio \
pytorch-cuda=11.8 kornia rich typer wandb -c pytorch -c nvidia -c fastai -c conda-forge