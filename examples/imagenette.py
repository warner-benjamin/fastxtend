from __future__ import annotations

import time
import inspect
import yaml
from enum import Enum

from typing import Optional

import typer
try:
    from rich import print
except ImportError:
    pass

from fastai.vision.all import *
try:
    import wandb
    from fastai.callback.wandb import *
    WANDB = True
except ImportError:
    WANDB = False

from fastxtend.vision.all import *
try:
    from fastxtend.ffcv.all import *
    FFCV = True
except ImportError:
    FFCV = False

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]})

imagenette_stats = ([0.465,0.458,0.429],[0.285,0.280,0.301])
imagewoof_stats  = ([0.496,0.461,0.399],[0.257,0.249,0.258])

class ImagenetteSize(str, Enum):
    small  = 'small'
    medium = 'medium'
    full   = 'full'

class ResNetModel(str, Enum):
    resnet18       = 'resnet18'
    xresnet18      = 'xresnet18'
    xresnext18     = 'xresnext18'
    resnet34       = 'resnet34'
    xresnet34      = 'xresnet34'
    xresnext34     = 'xresnext34'
    xseresnext34   = 'xse_resnext34'
    xecaresnext34  = 'xeca_resnext34'
    resnet50       = 'resnet50'
    xresnet50      = 'xresnet50'
    xresnext50     = 'xresnext50'
    resnext50      = 'resnext50_32x4d'
    xseresnext50   = 'xse_resnext50'
    xecaresnext50  = 'xeca_resnext50'
    resnet101      = 'resnet50'
    xresnet101     = 'xresnet50'
    xresnext101    = 'xresnext50'
    resnext101_32  = 'resnext101_32x8d'
    resnext101_64  = 'resnext101_64x4d'
    xseresnext101  = 'xse_resnext50'
    xecaresnext101 = 'xeca_resnext50'

class Activation(str, Enum):
    relu      = 'ReLU'
    leakyrelu = 'LeakyReLU'
    silu      = 'SiLU'
    mish      = 'Mish'
    gelu      = 'GELU'

class Pooling(str, Enum):
    maxpool     = 'MaxPool'
    avgpool     = 'AvgPool'
    blurpool    = 'BlurPool'
    maxblurpool = 'MaxBlurPool'

class OptimizerChoice(str, Enum):
    adam   = 'adam'
    ranger = 'ranger'
    adan   = 'adan'
    lamb   = 'lamb'
    sgd    = 'sgd'
    lion   = 'lion'

class Scheduler(str, Enum):
    onecycle  = 'one_cycle'
    flatcos   = 'flat_cos'
    flatwarm  = 'flat_warmup'
    cosanneal = 'cos_anneal'

class WarmMode(str, Enum):
    epoch = 'epoch'
    pct   = 'pct'
    auto  = 'auto'

class WarmSched(str, Enum):
    SchedCos = 'SchedCos'
    SchedLin = 'SchedLin'

# from maxb2: https://github.com/tiangolo/typer/issues/86#issuecomment-996374166
def conf_callback(ctx: typer.Context, param: typer.CallbackParam, config: Optional[str] = None):
    if config is not None:
        typer.echo(f"Loading config file: {config}\n")
        try:
            with open(config, 'r') as f:    # Load config file
                conf = yaml.safe_load(f)
            ctx.default_map = ctx.default_map or {}  # Initialize the default map
            ctx.default_map.update(conf)   # Merge the config dict into default_map
        except Exception as ex:
            raise typer.BadParameter(str(ex))
    return config


def get_dataset(size:ImagenetteSize, imagenette:bool):
    if size==ImagenetteSize.small:
        path = URLs.IMAGENETTE_160 if imagenette else URLs.IMAGEWOOF_160
    elif size==ImagenetteSize.medium:
        path = URLs.IMAGENETTE_320 if imagenette else URLs.IMAGEWOOF_320
    else:
        path = URLs.IMAGENETTE if imagenette else URLs.IMAGEWOOF
    source = untar_data(path)

    dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                       splitter=GrandparentSplitter(valid_name='val'),
                       get_items=get_image_files, get_y=parent_label)

    return dblock.datasets(source)


@app.command(help='Create an Imagenette or ImageWoof FFCV dataset in ~/.cache/fastxtend/')
def create(
    size:ImagenetteSize=typer.Option(ImagenetteSize.medium, show_default=ImagenetteSize.medium.value, help="Dataset image size. small=160, medium=320, full=fullsize.", case_sensitive=False),
    imagenette:bool=typer.Option(True, "--imagenette/--imagewoof", help="Create Imagenette or ImageWoof dataset.", rich_help_panel="Dataset"),
    jpeg:bool=typer.Option(False, "--jpeg/--raw", help="Save images as JPEGs instead of RAW. Uses less space, but not identical to fastai dataset."),
    jpeg_quality:int=typer.Option(90, help="JPEG quality if --jpeg."),
    chunk_size:int=typer.Option(100, help="Number of chunks processed by each worker. Lower to use less memory."),
):
    create_ffcv_dataset(size=size, imagenette=imagenette, jpeg=jpeg, jpeg_quality=jpeg_quality, chunk_size=chunk_size)


def create_ffcv_dataset(size:ImagenetteSize=ImagenetteSize.medium, imagenette:bool=False,
                        jpeg:bool=False, jpeg_quality:int=90, chunk_size:int=100):

    fn_base = '.cache/fastxtend/imagenette' if imagenette else '.cache/fastxtend/imagewoof'

    if size==ImagenetteSize.medium:
        train_fn = Path.home()/f'{fn_base}_320_train.ffcv'
        valid_fn = Path.home()/f'{fn_base}_320_valid.ffcv'
    elif size==ImagenetteSize.small:
        train_fn = Path.home()/f'{fn_base}_160_train.ffcv'
        valid_fn = Path.home()/f'{fn_base}_160_valid.ffcv'
    else:
        train_fn = Path.home()/f'{fn_base}_train.ffcv'
        valid_fn = Path.home()/f'{fn_base}_valid.ffcv'

    Path(train_fn).parent.mkdir(exist_ok=True)
    dataset = get_dataset(size, imagenette)

    for ds, fn in zip([dataset.train, dataset.valid], [train_fn, valid_fn]):
        rgb_dataset_to_ffcv(ds, fn, write_mode='jpeg' if jpeg else 'raw',
                            jpeg_quality=jpeg_quality, chunk_size=chunk_size)


def get_aug_transforms(flip:bool=True, flip_vert:bool=False, max_rotate:float=10., min_zoom:float=1.,
                       max_zoom:float=1., max_lighting:float=0.2, max_warp:float=0.2, prob_affine:float=0.75,
                       prob_lighting:float=0.75, prob_saturation:float=0, max_saturation:float=0.2,
                       prob_hue:float=0, max_hue:float=0.2, prob_grayscale:float=0., prob_channeldrop:float=0.,
                       prob_erasing:float=0.):
    xtra_tfms = []
    if prob_saturation > 0:
        xtra_tfms.append(Saturation(max_lighting=max_saturation, p=prob_saturation))
    if prob_hue > 0:
        xtra_tfms.append(Hue(max_hue=max_hue, p=prob_hue))
    if prob_saturation > 0:
        xtra_tfms.append(Grayscale(p=prob_grayscale))
    if prob_channeldrop > 0:
        xtra_tfms.append(ChannelDrop(p=prob_channeldrop))
    if prob_erasing > 0:
        xtra_tfms.append(RandomErasingBatch(p=prob_erasing))
    if len(xtra_tfms) == 0:
        xtra_tfms = None

    return *aug_transforms(do_flip=flip, flip_vert=flip_vert, max_rotate=max_rotate, min_zoom=min_zoom,
                           max_zoom=max_zoom, max_lighting=max_lighting, max_warp=max_warp, p_affine=prob_affine,
                           p_lighting=prob_lighting, xtra_tfms=xtra_tfms),


def get_fastai_dls(size:int, bs:int, imagenette:bool=False, max_workers:int=16, center_crop:bool=True,
                   device:int|str|torch.device|None=None, flip:bool=True, flip_vert:bool=False,
                   max_rotate:float=10., min_zoom:float=1., max_zoom:float=1., max_lighting:float=0.2,
                   max_warp:float=0.2, prob_affine:float=0.75, prob_lighting:float=0.75,
                   prob_saturation:float=0, max_saturation:float=0.2, prob_hue:float=0, max_hue:float=0.2,
                   prob_grayscale:float=0., prob_channeldrop:float=0., prob_erasing:float=0.):
    if size<=144:
        path = URLs.IMAGENETTE_160 if imagenette else URLs.IMAGEWOOF_160
    elif size<=224:
        path = URLs.IMAGENETTE_320 if imagenette else URLs.IMAGEWOOF_320
    else:
        path = URLs.IMAGENETTE if imagenette else URLs.IMAGEWOOF
    source = untar_data(path)

    workers = min(max_workers, num_cpus())

    gpu_tfms = get_aug_transforms(flip=flip, flip_vert=flip_vert, max_rotate=max_rotate, min_zoom=min_zoom,
                                  max_zoom=max_zoom, max_lighting=max_lighting, max_warp=max_warp,
                                  prob_affine=prob_affine, prob_lighting=prob_lighting, prob_saturation=prob_saturation,
                                  max_saturation=max_saturation, prob_hue=prob_hue, max_hue=max_hue,
                                  prob_grayscale=prob_grayscale, prob_channeldrop=prob_channeldrop, prob_erasing=prob_erasing)

    stats = imagenette_stats if imagenette else imagewoof_stats
    batch_tfms = [IntToFloatTensor(), *gpu_tfms, Normalize.from_stats(*stats)]

    dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                       splitter=GrandparentSplitter(valid_name='val'),
                       get_items=get_image_files, get_y=parent_label,
                       item_tfms=[RandomResizedCrop(size, min_scale=0.35)],
                       batch_tfms=batch_tfms)

    dls = dblock.dataloaders(source, bs=bs, num_workers=workers, device=device)

    if center_crop:
        vblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                           splitter=GrandparentSplitter(valid_name='val'),
                           get_items=get_image_files, get_y=parent_label,
                           item_tfms=[Resize(size)],
                           batch_tfms=batch_tfms)

        vls = vblock.dataloaders(source, bs=bs*2, num_workers=workers, device=device)
        dls.valid = vls.valid
        vls.train = None

    return dls


def get_ffcv_dls(size:int, bs:int, imagenette:bool=False, item_transforms:bool=False, max_workers:int=16,
                 seed:int=42, batches_ahead:int=2, device:int|str|torch.device|None=None,
                 quasi_random:bool=False, flip:bool=True, flip_vert:bool=False, max_rotate:float=10.,
                 max_zoom:float=1., min_zoom:float=1., max_lighting:float=0.2, max_warp:float=0.2,
                 prob_affine:float=0.75, prob_lighting:float=0.75, prob_saturation:float=0,
                 max_saturation:float=0.2, prob_hue:float=0, max_hue:float=0.2, prob_grayscale:float=0.,
                 prob_channeldrop:float=0., prob_erasing:float=0.):

    workers = min(max_workers, num_cpus())
    if item_transforms:
        gpu_tfms = affine_transforms(do_flip=flip, flip_vert=flip_vert, max_rotate=max_rotate,
                                     min_zoom=min_zoom, max_zoom=max_zoom, max_warp=max_warp,
                                     p_affine=prob_affine)
    else:
        gpu_tfms = get_aug_transforms(flip=flip, flip_vert=flip_vert, max_rotate=max_rotate, min_zoom=min_zoom,
                                      max_zoom=max_zoom, max_lighting=max_lighting, max_warp=max_warp,
                                      prob_affine=prob_affine, prob_lighting=prob_lighting, prob_saturation=prob_saturation,
                                      max_saturation=max_saturation, prob_hue=prob_hue, max_hue=max_hue,
                                      prob_grayscale=prob_grayscale, prob_channeldrop=prob_channeldrop,
                                      prob_erasing=prob_erasing)

    stats = imagenette_stats if imagenette else imagewoof_stats
    batch_tfms = [IntToFloatTensor(), *gpu_tfms, Normalize.from_stats(*stats)]

    if item_transforms:
        train_pipe = [
            RandomResizedCropRGBImageDecoder(output_size=(size,size), scale=(0.35, 1)),
            fx.RandomLighting(prob=prob_lighting, prob_saturation=prob_saturation, max_saturation=max_saturation)
        ]
        if prob_hue > 0:
            train_pipe.append(fx.RandomHue(prob=prob_hue, max_hue=max_hue))
        if prob_grayscale > 0:
            train_pipe.append(fx.RandomGrayscale(prob=prob_grayscale))
        if prob_channeldrop > 0:
            train_pipe.append(fx.RandomChannelDrop(prob=prob_channeldrop))
        if prob_erasing > 0:
            fill_mean = tuple([np.clip(i*255, 0, 255) for i in stats[0]])
            fill_std = tuple([np.clip(i*255, 0, 255) for i in stats[1]])
            train_pipe.append(fx.RandomErasing(prob=prob_erasing, fill_mean=fill_mean, fill_std=fill_std))
        train_pipe.extend([fx.ToTensorImage(), fx.ToDevice()])
    else:
        train_pipe = [
            RandomResizedCropRGBImageDecoder(output_size=(size,size), scale=(0.35, 1)),
            fx.ToTensorImage(),
            fx.ToDevice()
        ]
    valid_pipe = [
        CenterCropRGBImageDecoder(output_size=(size,size), ratio=1),
        fx.ToTensorImage(),
        fx.ToDevice()
    ]

    loaders = {}
    fn_base = '.cache/fastxtend/imagenette'if imagenette else '.cache/fastxtend/imagewoof'
    for name in ['valid', 'train']:
        if size<=144:
            file = Path.home()/f'{fn_base}_160_{name}.ffcv'
            ds_size = ImagenetteSize.small
        elif size<=224:
            file = Path.home()/f'{fn_base}_320_{name}.ffcv'
            ds_size = ImagenetteSize.medium
        else:
            file = Path.home()/f'{fn_base}_{name}.ffcv'
            ds_size = ImagenetteSize.full

        if not Path(file).exists():
            print("Creating FFCV dataset")
            create_ffcv_dataset(ds_size, imagenette)

        label_pipe = [
            IntDecoder(), fx.ToTensorCategory(),
            fx.Squeeze(), fx.ToDevice()
        ]
        train = OrderOption.QUASI_RANDOM if quasi_random else OrderOption.RANDOM
        loaders[name] = Loader(file,
                               batch_size=bs if name=='train' else bs*2,
                               num_workers=workers,
                               order=train if name=='train' else OrderOption.SEQUENTIAL,
                               pipelines={'image': train_pipe if name=='train' else valid_pipe, 'label': label_pipe},
                               batch_tfms=batch_tfms,
                               batches_ahead=batches_ahead,
                               seed=seed,
                               device=device)
    return DataLoaders(loaders['train'], loaders['valid'])


@app.command(help='Train a ResNet or XResNet model on Imagenette in Mixed Precision on a single GPU with either a fastxtend+ffcv or fastai dataloader.')
def train(ctx:typer.Context, # Typer Context to grab config for --verbose and passing to WandB
    # Config file
    config:Optional[Path]=typer.Option(None, callback=conf_callback, is_eager=True, help="Relative path to YAML config file for setting options. Passing CLI options will supersede config options.", case_sensitive=False, rich_help_panel="Options"),
    verbose:bool=typer.Option(False, "--verbose/--quiet", help="Print non-default options, excluding Weights and Biases options.", rich_help_panel="Options"),
    # Model
    model:ResNetModel=typer.Option(ResNetModel.xresnext50, show_default=ResNetModel.xresnext50.value, help="ResNet model architecture to train.", case_sensitive=False, rich_help_panel="Model"),
    act_cls:Activation=typer.Option(Activation.relu, show_default=Activation.relu.value, help="Activation function for fastxtend XResNet. Not case sensitive.", case_sensitive=False, rich_help_panel="Model"),
    dropout:float=typer.Option(0, help="Dropout for fastxtend XResNet.", rich_help_panel="Model"),
    stoch_depth:float=typer.Option(0, help="Stochastic depth for fastxtend XResNet.", rich_help_panel="Model"),
    stem_pool:Pooling=typer.Option(Pooling.maxpool, show_default=Pooling.maxpool.value, help="Stem pooling layer for fastxtend XResNet. Not case sensitive.", case_sensitive=False, rich_help_panel="Model"),
    block_pool:Pooling=typer.Option(Pooling.avgpool, show_default=Pooling.avgpool.value, help="ResBlock pooling layer for fastxtend XResNet. Not case sensitive.", case_sensitive=False, rich_help_panel="Model"),
    # Optimizer
    optimizer:OptimizerChoice=typer.Option(OptimizerChoice.ranger, show_default=OptimizerChoice.ranger.value, help="Which optimizer to use. Make sure to set learning rate if changed.", case_sensitive=False, rich_help_panel="Optimizer"),
    weight_decay:Optional[float]=typer.Option(None, help="Weight decay for Optimizer. If None, use optimizer's default.", rich_help_panel="Optimizer"),
    decouple_wd:bool=typer.Option(True, "--true-wd/--l2-wd", help="Apply true (decoupled) weight decay or L2 regularization. Doesn't apply to Adan or Lion.", rich_help_panel="Optimizer"),
    fused_opt:bool=typer.Option(True, "--fused/--standard", help="Use faster For Each fused Optimizer or slower standard fastai Optimizer.", rich_help_panel="Optimizer"),
    mom:Optional[float]=typer.Option(None, help="Gradient moving average (β1) coefficient. If None, uses optimizer's default.", rich_help_panel="Optimizer"),
    sqr_mom:Optional[float]=typer.Option(None, help="Gradient squared moving average (β2) coefficient. If None, use optimizer's default.", rich_help_panel="Optimizer"),
    beta1:Optional[float]=typer.Option(None, help="Adan: Gradient moving average (β1) coefficient. Lion: Update gradient moving average (β1) coefficient. If None, use optimizer's default.", rich_help_panel="Optimizer"),
    beta2:Optional[float]=typer.Option(None, help="Adan: Gradient difference moving average (β2) coefficient. Lion: Gradient moving average (β2) coefficient. If None, use optimizer's default.", rich_help_panel="Optimizer"),
    beta3:Optional[float]=typer.Option(None, help="Adan: Gradient squared moving average (β3) coefficient. If None, use optimizer's default.", rich_help_panel="Optimizer"),
    eps:Optional[float]=typer.Option(None, help="Added for numerical stability. If None, uses optimizer's default.", rich_help_panel="Optimizer"),
    paper_init:bool=typer.Option(False, "--paperinit/--zeroinit", help="Adan: Initialize prior gradient with current gradient per paper or zeroes.", rich_help_panel="Optimizer"),
    # Scheduler
    scheduler:Scheduler=typer.Option(Scheduler.flatcos, show_default=Scheduler.flatcos.value, help="Which fastai or fastxtend scheduler to use. fit_one_cycle, fit_flat_cos, fit_flat_warmup, or fit_cos_anneal.", case_sensitive=False, rich_help_panel="Scheduler"),
    epochs:int=typer.Option(20, help="Number of epochs to train for.", rich_help_panel="Scheduler"),
    learning_rate:float=typer.Option(8e-3, help="Max Learning rate for the scheduler. Make sure to change if --optimizer is changed.", rich_help_panel="Scheduler"),
    pct_start:Optional[float]=typer.Option(None, help="Scheduler's percent start. Uses scheduler's default if None.", rich_help_panel="Scheduler"),
    warm_pct:float=typer.Option(0.2, help="Learning rate warmup in percent of training steps. Only applies to cos_warmup or cos_anneal.", rich_help_panel="Scheduler"),
    warm_epoch:int=typer.Option(5, help="Learning rate warmup in training epochs. Only applies to cos_warmup or cos_anneal.", rich_help_panel="Scheduler"),
    warm_mode:WarmMode=typer.Option(WarmMode.auto, show_default=WarmMode.auto.value, help="Warmup using 'epoch', 'pct', or min of epoch/pct if 'auto'. Only applies to cos_warmup or cos_anneal.", rich_help_panel="Scheduler"),
    warm_sched:WarmSched=typer.Option(WarmSched.SchedCos, show_default=WarmMode.auto.value, help="Learning rate warmup schedule. Not case sensitive.", case_sensitive=False, rich_help_panel="Scheduler"),
    div_start:float=typer.Option(0.25, help="# Initial learning rate: `lr/div_start`.", rich_help_panel="Scheduler"),
    div_final:float=typer.Option(1e5, help="Final learning rate: `lr/div_final`.", rich_help_panel="Scheduler"),
    # Training
    label_smoothing:float=typer.Option(0.1, help="nn.CrossEntropyLoss label_smoothing amount.", rich_help_panel="Training"),
    seed:int=typer.Option(42, help="Random seed to use. Note: fastxtend+ffcv item transforms are not seeded.", rich_help_panel="Training"),
    channels_last:bool=typer.Option(True, "--channels-last/--fp16", help="Train in channels last format or Mixed Precision. Channels Last equires a modern GPU (RTX 2000, Turing, Volta, or newer).", rich_help_panel="Training"),
    profile:bool=typer.Option(False, "--profile", help="Profile training speed using fastxtend's Throughput profiler.", rich_help_panel="Training"),
    # Dataset
    image_size:int=typer.Option(224, help="Image size. Automatically downloads and creates dataset if it doesn't exist.", rich_help_panel="Dataset"),
    batch_size:int=typer.Option(64, help="Batch size. Increase learning rate to 1e-2 (for ranger) if batch size is 128.", rich_help_panel="Dataset"),
    imagenette:bool=typer.Option(True, "--imagenette/--imagewoof", help="Train on Imagenette or ImageWoof", rich_help_panel="Dataset"),
    # Progressive Resizing
    prog_resize:bool=typer.Option(True, "--prog-resize/--full-size", help="Use the automatic Progressive Resizing callback. Significantly faster. May need to train for a few more epochs for same accuracy.", rich_help_panel="Progressive Resizing"),
    increase_by:int=typer.Option(16, help="Pixel increase amount for resizing step. 16 is good for 20-25 epochs. Requires passing --prog-resize.", rich_help_panel="Progressive Resizing"),
    initial_size:float=typer.Option(0.5, help="Staring size relative to --size. Requires passing --prog-resize.", rich_help_panel="Progressive Resizing"),
    resize_start:float=typer.Option(0.5, help="Earliest upsizing epoch in percent of training time. Requires passing --prog-resize.", rich_help_panel="Progressive Resizing"),
    resize_finish:float=typer.Option(0.75, help="Last upsizing epoch in percent of training time. Requires passing --prog-resize.", rich_help_panel="Progressive Resizing"),
    preallocate:bool=typer.Option(False, "--preallocate", help="Preallocate GPU memory by performing a dry run at final size. May prevent stuttering during training due to memory allocation. Requires passing --prog-resize.", rich_help_panel="Progressive Resizing"),
    # Dataloader
    use_ffcv:bool=typer.Option(True, "--ffcv/--fastai", help="Use fastxtend+ffcv dataloader or the fastai dataloader. fastxtend+ffcv can be significantly faster.", rich_help_panel="DataLoader"),
    max_workers:int=typer.Option(16, help="Maximum number of workers to use for multiprocessing. Chooses number of CPUs if lower.", rich_help_panel="DataLoader"),
    device:Optional[str]=typer.Option(None, help="Device to train on. If not set, uses the fastai default device. Must be Cuda device if --ffcv.", rich_help_panel="DataLoader"),
    center_crop:bool=typer.Option(True, "--crop/--squish", help="Center crop or squish validation images with the fastai dataloader. --crop matches fastxtend+ffcv dataloader.", rich_help_panel="DataLoader"),
    # FFCV Dataloader
    item_transforms:bool=typer.Option(False, "--item-tfms/--batch-tfms", help="Where possible, use fastxtend+ffcv Numba compliled item transforms instead of GPU batch transforms.", rich_help_panel="fastxtend+ffcv DataLoader"),
    batches_ahead:int=typer.Option(1, help="Number of batches prepared in advance by fastxtend+ffcv dataloader. Balances latency and memory usage.", rich_help_panel="fastxtend+ffcv DataLoader"),
    quasi_random:bool=typer.Option(False, "--random/--quasi", help="Use Random or Quasi-Random loading with fastxtend+ffcv dataloader. Quasi-Random is for a low memory machine.", rich_help_panel="fastxtend+ffcv DataLoader"),
    # Transform Options
    flip:bool=typer.Option(True, help="Randomly flip the image horizontally", rich_help_panel="Transform Options"),
    flip_vert:bool=typer.Option(False, help="Randomly flip the image vertically", rich_help_panel="Transform Options"),
    max_rotate:float=typer.Option(10., help="Maximum degrees of rotation to apply to the image", rich_help_panel="Transform Options"),
    min_zoom:float=typer.Option(1., help="Minimum zoom level to apply to the image", rich_help_panel="Transform Options"),
    max_zoom:float=typer.Option(1., help="Maximum zoom level to apply to the image", rich_help_panel="Transform Options"),
    max_lighting:float=typer.Option(0.2, help="Maximum scale of brightness and contrast to apply to the image", rich_help_panel="Transform Options"),
    max_warp:float=typer.Option(0.2, help="Maximum value of warp per to apply to the image", rich_help_panel="Transform Options"),
    prob_affine:float=typer.Option(0.75, help="Probability of applying affine transformation to the image", rich_help_panel="Transform Options"),
    prob_lighting:float=typer.Option(0.75, help="Probability of changing the brightness and contrast of the image", rich_help_panel="Transform Options"),
    prob_saturation:float=typer.Option(0, help="Probability of changing the saturation of the image. If 0, is not applied.", rich_help_panel="Transform Options"),
    max_saturation:float=typer.Option(0.2, help="Maximum scale of saturationto apply to the image.", rich_help_panel="Transform Options"),
    prob_hue:float=typer.Option(0, help="Probability of changing the hue of the image. If 0, is not applied.", rich_help_panel="Transform Options"),
    max_hue:float=typer.Option(0.2, help="Maximum hue factor to apply to the image.", rich_help_panel="Transform Options"),
    prob_grayscale:float=typer.Option(0, help="Probability of changing the image to grayscale. If 0, is not applied.", rich_help_panel="Transform Options"),
    prob_channeldrop:float=typer.Option(0, help="Probability of replacing one channel with a single value. If 0, is not applied.", rich_help_panel="Transform Options"),
    prob_erasing:float=typer.Option(0, help="Probability of applying random erasing on an image. If 0, is not applied.", rich_help_panel="Transform Options"),
    # MixUp & CutMix
    mixup:bool=typer.Option(False, "--mixup", help="Train with MixUp. Only use if training for more than 60 epochs.", rich_help_panel="MixUp & CutMix"),
    mixup_alpha:float=typer.Option(0.4, help="MixUp's Alpha & beta parametrization for Beta distribution. Requires passing --mixup or --cutmixup.", rich_help_panel="MixUp & CutMix"),
    cutmix:bool=typer.Option(False, "--cutmix", help="Train in CutMix. Only use if training for more than 60 epochs.", rich_help_panel="MixUp & CutMix"),
    cutmix_alpha:float=typer.Option(1, help="CutMix's Alpha & beta parametrization for Beta distribution. Requires passing --cutmix or --cutmixup.", rich_help_panel="MixUp & CutMix"),
    cutmixup:bool=typer.Option(False, "--cutmixup", help="Train with CutMix and MixUp. Only use if training for more than 60 epochs.", rich_help_panel="MixUp & CutMix"),
    mixup_ratio:float=typer.Option(1, help="MixUp ratio relative to CutMix. Requires passing --cutmixup.", rich_help_panel="MixUp & CutMix"),
    cutmix_ratio:float=typer.Option(1, help="CutMix ratio relative to MixUp. Requires passing --cutmixup.", rich_help_panel="MixUp & CutMix"),
    elementwise:bool=typer.Option(False, "--element-cutmixup/--batch-cutmixup", help="Elementwise CutMix and Mixup on the same batch or one per entire batch. Requires passing --cutmixup.", rich_help_panel="MixUp & CutMix"),
    # Weights and Biases
    log_wandb:bool=typer.Option(False, "--wandb", help="Log to Weights and Biases.", rich_help_panel="Weights and Biases"),
    name:Optional[str]=typer.Option(None, help="WandB run name.", rich_help_panel="Weights and Biases"),
    project:Optional[str]=typer.Option(None, help="WandB project name.", rich_help_panel="Weights and Biases"),
    group:Optional[str]=typer.Option(None, help="WandB group name.", rich_help_panel="Weights and Biases"),
    tags:Optional[str]=typer.Option(None, help="WandB tags. String of comma seperated values.", rich_help_panel="Weights and Biases"),
    entity:Optional[str]=typer.Option(None, help="WandB entity name.", rich_help_panel="Weights and Biases"),
    save_code:Optional[bool]=typer.Option(False, "--save-code", help="Save code to WandB. Requires passing --wandb", rich_help_panel="Weights and Biases"),
):
    # Create passed run config from optional loaded config file, option defaults, and passed options.
    # Ignoring the wandb options.
    ignore_params = ['config', 'verbose', 'log_wandb', 'name', 'project', 'group', 'tags', 'entity', 'save_code']
    config = {k:v for k,v in ctx.params.items() if k not in ignore_params}

    # Print non-default options
    if verbose:
        non_defaults = {}
        orig_params = inspect.signature(train).parameters
        for k,v in config.items():
            dv = orig_params[k].default.default
            if v != dv:
                non_defaults[k] = v
        if len(non_defaults.keys()) > 0:
            print("Modified Parameters:")
            print(non_defaults)
            print()

    if not FFCV and use_ffcv:
        msg = f"{use_ffcv=} passed, but ffcv not found. Install ffcv with `pip install fastxtend[ffcv]`"\
               " or see docs for conda installation (recommended)."
        raise ImportError(msg)

    if profile:
        from fastxtend.callback import profiler

    # Grab model from global imports and test if fastai with n_out or pytorch with num_classes
    model = globals()[model.value]
    act_cls = getattr(nn, act_cls.value)
    stem_pool = globals()[stem_pool.value]
    block_pool = globals()[block_pool.value]

    if pct_start is None:
        pct_start = 0.25 if scheduler==Scheduler.onecycle else 0.75

    # If n_out exists, its a XResNet model
    if 'n_out' in inspect.getfullargspec(model).args:
        arch = partial(model, n_out=10, act_cls=act_cls, p=dropout, stoch_depth=stoch_depth,
                       stem_pool=stem_pool, block_pool=block_pool)
    else:
        arch = partial(model, num_classes=10)

    # Grab optimizer from global imports, update opt_kwargs from passed non-default options
    opt = globals()[optimizer.value]
    opt_params = inspect.signature(opt).parameters
    opt_kwargs = {k:v for k,v in config.items() if k in opt_params.keys() and v is not None}
    if 'foreach' in opt_kwargs.keys():
        opt_kwargs.pop('foreach')

    # Add any supported callbacks and their options
    cbs = []
    if prog_resize:
        cbs += [ProgressiveResize(initial_size=initial_size, start=resize_start,
                                  finish=resize_finish, increase_by=increase_by,
                                  preallocate=preallocate)]
    if mixup:
        cbs += [MixUp(mixup_alpha=mixup_alpha, interp_label=False)]
    elif cutmix:
        cbs += [CutMix(cutmix_alpha=cutmix_alpha, interp_label=False)]
    elif cutmixup:
        cbs += [CutMixUp(mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, mixup_ratio=mixup_ratio,
                         cutmix_ratio=cutmix_ratio, element=elementwise, interp_label=False)]

    # Create the dataloaders
    with less_random(seed):
        if use_ffcv:
            dls = get_ffcv_dls(size=image_size, bs=batch_size, imagenette=imagenette,
                               item_transforms=item_transforms, max_workers=max_workers, seed=seed,
                               batches_ahead=batches_ahead, device=device, quasi_random=quasi_random, flip=flip,
                               flip_vert=flip_vert, max_rotate=max_rotate, min_zoom=min_zoom, max_zoom=max_zoom,
                               max_lighting=max_lighting, max_warp=max_warp, prob_affine=prob_affine,
                               prob_lighting=prob_lighting, prob_saturation=prob_saturation,
                               max_saturation=max_saturation, prob_hue=prob_hue, max_hue=max_hue,
                               prob_grayscale=prob_grayscale, prob_channeldrop=prob_channeldrop,
                               prob_erasing=prob_erasing)
        else:
            dls = get_fastai_dls(size=image_size, bs=batch_size, imagenette=imagenette, max_workers=max_workers,
                                 center_crop=center_crop, device=device, flip=flip, flip_vert=flip_vert,
                                 max_rotate=max_rotate, min_zoom=min_zoom, max_zoom=max_zoom, max_lighting=max_lighting,
                                 max_warp=max_warp, prob_affine=prob_affine, prob_lighting=prob_lighting,
                                 prob_saturation=prob_saturation, max_saturation=max_saturation, prob_hue=prob_hue,
                                 max_hue=max_hue, prob_grayscale=prob_grayscale, prob_channeldrop=prob_channeldrop,
                                 prob_erasing=prob_erasing)

    # Setup Weights and Biases logging, if initalized
    if log_wandb:
        if not WANDB:
            raise ImportError(f"{log_wandb=} passed, but wandb not found. Install wandb with `pip install wandb`.")
        wandb.init(name=name, project=project, group=group,
                   tags=tags.split(",") if isinstance(tags, str) else None,
                   entity=entity, save_code=save_code, config=config)
        cbs += [WandbCallback(log_preds=False)]

    # Create Learner
    with less_random(seed):
        learn = Learner(dls, arch(), loss_func=nn.CrossEntropyLoss(label_smoothing=label_smoothing),
                        opt_func=opt(foreach=fused_opt, **opt_kwargs), metrics=Accuracy(), cbs=cbs)
        learn.to_channelslast() if channels_last else learn.to_fp16()
        if profile:
            learn.profile()

    # Train
    start = time.perf_counter()
    with less_random(seed):
        if scheduler==Scheduler.flatcos:
            learn.fit_flat_cos(n_epoch=epochs, lr=learning_rate, wd=weight_decay,
                               pct_start=pct_start, div_final=div_final)
        elif scheduler==Scheduler.onecycle:
            learn.fit_one_cycle(n_epoch=epochs, lr=learning_rate, wd=weight_decay,
                                pct_start=pct_start, div=div_start, div_final=div_final)
        elif scheduler==Scheduler.flatwarm:
            learn.fit_flat_warm(n_epoch=epochs, lr=learning_rate, wd=weight_decay,
                                pct_start=pct_start, div=div_start, div_final=div_final,
                                warm_pct=warm_pct, warm_epoch=warm_epoch,
                                warm_mode=warm_mode.value, warm_sched=globals()[warm_sched.value])
        elif scheduler==Scheduler.cosanneal:
            learn.fit_cos_anneal(n_epoch=epochs, lr=learning_rate, wd=weight_decay,
                                 pct_start=pct_start, div=div_start, div_final=div_final,
                                 warm_pct=warm_pct, warm_epoch=warm_epoch,
                                 warm_mode=warm_mode.value, warm_sched=globals()[warm_sched.value])
    end = time.perf_counter()
    print(f"\nTotal training time: {end-start:.2f} seconds\n")


if __name__=="__main__":
    app()