conda create -n fastxtend python=3.11 "pytorch>=2.1" torchvision torchaudio \
pytorch-cuda=12.1 fastai nbdev pkg-config libjpeg-turbo opencv tqdm psutil \
terminaltables numpy "numba>=0.57" librosa timm kornia rich typer wandb \
"transformers>=4.34" "tokenizers>=0.14" "datasets>=2.14" ipykernel ipywidgets \
"matplotlib<3.8" -c pytorch -c nvidia -c fastai -c huggingface -c conda-forge