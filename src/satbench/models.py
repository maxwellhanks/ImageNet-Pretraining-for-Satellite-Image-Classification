"""Model registry.

Each model is registered under a name and returns a ``ModelBundle``: the network
plus its associated PyTorch data transforms (train/eval). Keeping the transforms
tied to the model ensures that pre-trained weights always receive the exact 
normalisation statistics they expect.

Add a model:

    @register_model("my_model")
    def _my_model(num_classes: int, run: RunConfig) -> ModelBundle:
        net = ...
        train_tf = transforms.Compose([...])
        eval_tf = transforms.Compose([...])
        return ModelBundle(net, train_tf, eval_tf, "description")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16
from torchvision import transforms

from .config import RunConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class ModelBundle:
    """Pairs a network with its required data augmentations and preprocessing."""
    net: nn.Module
    train_transform: transforms.Compose
    eval_transform: transforms.Compose
    description: str

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Returns only the parameters that require gradients."""
        return [p for p in self.net.parameters() if p.requires_grad]

    def n_params(self) -> tuple[int, int]:
        total = sum(p.numel() for p in self.net.parameters())
        return total, sum(p.numel() for p in self.trainable_parameters())


ModelBuilder = Callable[[int, RunConfig], ModelBundle]
MODELS: dict[str, ModelBuilder] = {}


def register_model(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    """Decorator to register a new model builder function under the given name."""
    def deco(fn: ModelBuilder) -> ModelBuilder:
        if name in MODELS:
            raise KeyError(f"Model '{name}' registered twice")
        MODELS[name] = fn
        return fn
    return deco


def build_model(name: str, num_classes: int, run: RunConfig) -> ModelBundle:
    """Instantiates a model and its associated transforms from the registry."""
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODELS)}")
    return MODELS[name](num_classes, run)


@register_model("linear")
def _linear(num_classes: int, run: RunConfig) -> ModelBundle:
    """Multinomial logistic regression on raw pixels at native 64x64 resolution.
    (Upsampling to 224 would add 12x more weights and no information.)"""
    size = 64
    net = nn.Sequential(nn.Flatten(), nn.Linear(3 * size * size, num_classes))
    train_tf = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Resize((size, size), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    eval_tf = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Resize((size, size), antialias=True),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    return ModelBundle(net, train_tf, eval_tf,
                       "Linear classifier on raw RGB pixels (64x64)")


SSL4EO_S12_RESNET50_MOCO_URL = (
    "https://huggingface.co/torchgeo/resnet50_sentinel2_rgb_moco/"
    "resolve/main/resnet50_sentinel2_rgb_moco-2b57ba8b.pth"
)


def _load_ssl4eo_weights(net: nn.Module) -> None:
    """Downloads and loads SSL4EO-S12 MoCo-v2 Sentinel-2 RGB ResNet-50 weights."""
    state_dict = torch.hub.load_state_dict_from_url(
        SSL4EO_S12_RESNET50_MOCO_URL,
        map_location="cpu",
        progress=True,
    )
    net.load_state_dict(state_dict, strict=False)


def _resnet50(
    num_classes: int,
    weights: ResNet50_Weights | str | None,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Helper to build a ResNet-50 and replace its final classification head."""
    if weights == "ssl4eo_s12":
        net = resnet50(weights=None)
        _load_ssl4eo_weights(net)
    else:
        net = resnet50(weights=weights)

    # freeze everything
    if freeze_backbone:
        for param in net.parameters():
            param.requires_grad = False

    # add new randomly initialized head
    net.fc = nn.Linear(net.fc.in_features, num_classes)

    return net

def _resnet_transforms(size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Returns standard train/eval ImageNet-style transforms with basic satellite augmentations."""
    train_tf = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Resize((size, size), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    eval_tf = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Resize((size, size), antialias=True),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    return train_tf, eval_tf


# EuroSAT RGB /255 -> x * SSL4EO_INPUT_SCALE (see _ssl4eo_transforms)
SSL4EO_INPUT_SCALE = 0.5


def _ssl4eo_transforms(size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Returns transforms matching the SSL4EO-S12 MoCo pre-training input range.

    SSL4EO-S12 was pre-trained on an 8-bit version of the data: per band,
    Sentinel-2 L1C values were stretched from [mean - 2 std, mean + 2 std] to
    [0, 255] and fed through ``ToTensor`` (/255), i.e. inputs in [0, 1] with
    *no* ImageNet mean/std normalisation. (torchgeo's ``Normalize(0, 10000)``
    is for raw uint16 reflectance and is wrong for 8-bit EuroSAT JPEGs.)

    EuroSAT RGB JPEGs use a brighter rendering than that stretch, so /255
    alone overshoots. Scaling by 0.5 best matches the pretrained BatchNorm
    running statistics and gives the best frozen-feature linear probe
    (EuroSAT val, 2k/2k images: ImageNet norm 74.1%, /10000 90.5%,
    /255 94.5%, /255 * 0.5 95.1%).
    """
    scale = transforms.Normalize((0.0, 0.0, 0.0), (1 / SSL4EO_INPUT_SCALE,) * 3)
    train_tf = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Resize((size, size), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        scale,
    ])
    eval_tf = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Resize((size, size), antialias=True),
        scale,
    ])
    return train_tf, eval_tf


@register_model("resnet50_random")
def _resnet50_random(num_classes: int, run: RunConfig) -> ModelBundle:
    train_tf, eval_tf = _resnet_transforms(run.image_size)
    return ModelBundle(_resnet50(num_classes, None),
                       train_tf, eval_tf,
                       "ResNet-50, random init, all layers trained")

@register_model("resnet50_random_linear_probe")
def _resnet50_random_linear_probe(num_classes: int, run: RunConfig) -> ModelBundle:
    train_tf, eval_tf = _resnet_transforms(run.image_size)
    return ModelBundle(_resnet50(num_classes, None, freeze_backbone=True),
                       train_tf, eval_tf,
                       "ResNet-50, random init, linear probed")


@register_model("resnet50_imagenet")
def _resnet50_imagenet(num_classes: int, run: RunConfig) -> ModelBundle:
    train_tf, eval_tf = _resnet_transforms(run.image_size)
    return ModelBundle(_resnet50(num_classes, ResNet50_Weights.IMAGENET1K_V1),
                       train_tf, eval_tf,
                       "ResNet-50, ImageNet-1k V1 init, all layers fine-tuned")

@register_model("resnet50_imagenet_linear_probe")
def _resnet50_imagenet_linear_probe(num_classes: int, run: RunConfig) -> ModelBundle:
    train_tf, eval_tf = _resnet_transforms(run.image_size)
    return ModelBundle(_resnet50(num_classes, ResNet50_Weights.IMAGENET1K_V1, freeze_backbone=True),
                       train_tf, eval_tf,
                       "ResNet-50, ImageNet-1k V1 init, linear probe")


@register_model("resnet50_ssl4eo")
def _resnet50_ssl4eo(num_classes: int, run: RunConfig) -> ModelBundle:
    train_tf, eval_tf = _ssl4eo_transforms(run.image_size)
    return ModelBundle(_resnet50(num_classes, "ssl4eo_s12"),
                       train_tf, eval_tf,
                       "ResNet-50, SSL4EO-S12 MoCo init, all layers fine-tuned")

@register_model("resnet50_ssl4eo_linear_probe")
def _resnet50_ssl4eo_linear_probe(num_classes: int, run: RunConfig) -> ModelBundle:
    train_tf, eval_tf = _ssl4eo_transforms(run.image_size)
    return ModelBundle(_resnet50(num_classes, "ssl4eo_s12", freeze_backbone=True),
                       train_tf, eval_tf,
                       "ResNet-50, SSL4EO-S12 MoCo init, linear probe")


def _vit_b_16(
    num_classes: int,
    weights: ViT_B_16_Weights | None,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Helper to build a ViT-B/16 and replace its final classification head."""
    net = vit_b_16(weights=weights)

    if freeze_backbone:
        for param in net.parameters():
            param.requires_grad = False

    # torchvision ViT uses ``heads.head`` (not ``fc`` like ResNet)
    net.heads.head = nn.Linear(net.heads.head.in_features, num_classes)
    return net


@register_model("vit_b16_random")
def _vit_b16_random(num_classes: int, run: RunConfig) -> ModelBundle:
    train_tf, eval_tf = _resnet_transforms(run.image_size)
    return ModelBundle(_vit_b_16(num_classes, None),
                       train_tf, eval_tf,
                       "ViT-B/16, random init, all layers trained")

@register_model("vit_b16_random_linear_probe")
def _vit_b16_random_linear_probe(num_classes: int, run: RunConfig) -> ModelBundle:
    train_tf, eval_tf = _resnet_transforms(run.image_size)
    return ModelBundle(_vit_b_16(num_classes, None, freeze_backbone=True),
                       train_tf, eval_tf,
                       "ViT-B/16, random init, linear probe")


@register_model("vit_b16_imagenet")
def _vit_b16_imagenet(num_classes: int, run: RunConfig) -> ModelBundle:
    train_tf, eval_tf = _resnet_transforms(run.image_size)
    return ModelBundle(_vit_b_16(num_classes, ViT_B_16_Weights.IMAGENET1K_V1),
                       train_tf, eval_tf,
                       "ViT-B/16, ImageNet-1k V1 init, all layers fine-tuned")

@register_model("vit_b16_imagenet_linear_probe")
def _vit_b16_imagenet_linear_probe(num_classes: int, run: RunConfig) -> ModelBundle:
    train_tf, eval_tf = _resnet_transforms(run.image_size)
    return ModelBundle(
        _vit_b_16(num_classes, ViT_B_16_Weights.IMAGENET1K_V1, freeze_backbone=True),
        train_tf, eval_tf,
        "ViT-B/16, ImageNet-1k V1 init, linear probe",
    )
