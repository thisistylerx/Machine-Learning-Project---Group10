import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def md_cell(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip("\n").splitlines()],
    }


def code_cell(code: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip("\n").splitlines()],
    }


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_IMPORT = """
from pathlib import Path
import json
import os
import random
import sys

from scripts.project_utils import (
    ARTIFACTS,
    DATA_ROOT,
    ETIS_ROOT,
    PVT_PRETRAINED_ROOT,
    ensure_project_dirs,
    get_default_project_config,
    load_project_config,
    print_env_summary,
    save_json,
    set_seed,
    try_import_torch,
)

ensure_project_dirs()
set_seed()
torch = try_import_torch()
ENV_SUMMARY = print_env_summary(torch)
ROOT = Path.cwd().resolve()
"""


CONFIG_BLOCK = """
PROJECT_CONFIG = get_default_project_config()
save_json(PROJECT_CONFIG, ARTIFACTS / "project_config.json")
PROJECT_CONFIG
"""


CONFIG_LOAD_BLOCK = """
PROJECT_CONFIG = load_project_config()
PROJECT_CONFIG
"""


ETIS_BOOTSTRAP_BLOCK = """
assert torch is not None, "需要先安装 PyTorch 才能运行本 notebook。"

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

ETIS_SPLITS = {
    "train": ETIS_ROOT / "train",
    "val": ETIS_ROOT / "val",
    "test": ETIS_ROOT / "test",
}

def read_list_file(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def collect_split_summary(split_name):
    split_root = ETIS_SPLITS[split_name]
    image_dir = split_root / "images"
    mask_dir = split_root / "masks"
    images = sorted(p.name for p in image_dir.glob("*") if p.is_file())
    masks = sorted(p.name for p in mask_dir.glob("*") if p.is_file())
    list_file = ETIS_ROOT / f"{split_name}_list_etis.txt"
    listed = read_list_file(list_file)
    return {
        "split": split_name,
        "image_count": len(images),
        "mask_count": len(masks),
        "list_count": len(listed),
        "image_mask_match": images == masks,
        "list_match_images": listed == images,
        "preview": images[:8],
    }

def infer_fixed_visual_sample():
    test_items = read_list_file(ETIS_ROOT / "test_list_etis.txt")
    return test_items[0] if test_items else None

def load_rgb_mask_pair(split_name, filename):
    image_path = ETIS_SPLITS[split_name] / "images" / filename
    mask_path = ETIS_SPLITS[split_name] / "masks" / filename
    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))
    return image, mask

def visualize_pair(split_name, filename):
    image, mask = load_rgb_mask_pair(split_name, filename)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image)
    axes[0].set_title(f"{split_name} image")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("mask")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig
"""


ETIS_SHARED_PIPELINE = """
assert torch is not None, "需要先安装 PyTorch 才能运行本 notebook。"

import math
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

TRAIN_IMAGE_DIR = ETIS_ROOT / "train" / "images"
TRAIN_MASK_DIR = ETIS_ROOT / "train" / "masks"
VAL_IMAGE_DIR = ETIS_ROOT / "val" / "images"
VAL_MASK_DIR = ETIS_ROOT / "val" / "masks"
TEST_IMAGE_DIR = ETIS_ROOT / "test" / "images"
TEST_MASK_DIR = ETIS_ROOT / "test" / "masks"
TRAIN_LIST_PATH = ETIS_ROOT / "train_list_etis.txt"
VAL_LIST_PATH = ETIS_ROOT / "val_list_etis.txt"
TEST_LIST_PATH = ETIS_ROOT / "test_list_etis.txt"
PVT_B0_PATH = PVT_PRETRAINED_ROOT / "pvt_v2_b0.pth"

def read_list_file(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def default_visual_sample():
    items = read_list_file(TEST_LIST_PATH)
    return items[0] if items else None

def load_rgb_mask(image_path, mask_path, image_size):
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    mask = Image.open(mask_path).convert("L").resize((image_size, image_size), Image.NEAREST)
    image = np.array(image, dtype=np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    mask = (np.array(mask, dtype=np.float32) > 127).astype(np.float32)
    image = torch.from_numpy(image.transpose(2, 0, 1))
    mask = torch.from_numpy(mask).unsqueeze(0)
    return image, mask

class ETISSegmentationDataset(Dataset):
    def __init__(self, split, image_size=352):
        self.split = split
        self.image_size = image_size
        self.image_dir = ETIS_ROOT / split / "images"
        self.mask_dir = ETIS_ROOT / split / "masks"
        self.items = read_list_file(ETIS_ROOT / f"{split}_list_etis.txt")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        filename = self.items[idx]
        image_path = self.image_dir / filename
        mask_path = self.mask_dir / filename
        image, mask = load_rgb_mask(image_path, mask_path, self.image_size)
        return {
            "name": filename,
            "image": image,
            "mask": mask,
        }

def build_dataloaders(cfg):
    train_dataset = ETISSegmentationDataset("train", image_size=cfg["image_size"])
    val_dataset = ETISSegmentationDataset("val", image_size=cfg["image_size"])
    test_dataset = ETISSegmentationDataset("test", image_size=cfg["image_size"])
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=cfg["num_workers"])
    return train_loader, val_loader, test_loader

def dice_from_logits(logits, mask, eps=1e-6):
    prediction = (torch.sigmoid(logits) > 0.5).float()
    intersection = (prediction * mask).sum(dim=(1, 2, 3))
    union = prediction.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3))
    score = (2 * intersection + eps) / (union + eps)
    return score.mean()

class DiceLoss(nn.Module):
    def forward(self, logits, mask):
        probability = torch.sigmoid(logits)
        intersection = (probability * mask).sum(dim=(1, 2, 3))
        union = probability.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3))
        loss = 1 - (2 * intersection + 1e-6) / (union + 1e-6)
        return loss.mean()

class SegmentationCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, mask):
        return self.bce(logits, mask) + self.dice(logits, mask)

def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_dice = 0.0
    steps = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item())
        total_dice += float(dice_from_logits(logits, masks).item())
        steps += 1
    return {
        "loss": total_loss / max(steps, 1),
        "dice": total_dice / max(steps, 1),
    }

def train_model(model, train_loader, val_loader, cfg, device="cpu"):
    criterion = SegmentationCriterion()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    best_state = None
    best_dice = -1.0
    history = []
    for epoch in range(cfg["epochs"]):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer=optimizer, device=device)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer=None, device=device)
        record = {
            "epoch": epoch + 1,
            "train_loss": round(train_metrics["loss"], 4),
            "train_dice": round(train_metrics["dice"], 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_dice": round(val_metrics["dice"], 4),
        }
        history.append(record)
        print(record)
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, round(best_dice, 4)

def evaluate_loader(model, loader, device="cpu"):
    criterion = SegmentationCriterion()
    summary = run_epoch(model, loader, criterion, optimizer=None, device=device)
    summary = {
        "loss": round(summary["loss"], 4),
        "dice": round(summary["dice"], 4),
    }
    per_sample = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            logits = model(image)
            per_sample.append(
                {
                    "name": batch["name"][0],
                    "dice": round(float(dice_from_logits(logits, mask).item()), 4),
                }
            )
    return summary, per_sample

def export_visualization(model, sample_name, image_size, save_path, device="cpu"):
    image_path = TEST_IMAGE_DIR / sample_name
    mask_path = TEST_MASK_DIR / sample_name
    image, mask = load_rgb_mask(image_path, mask_path, image_size)
    model.eval()
    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
        prediction = (torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy() > 0.5).astype(np.float32)
    image_np = image.permute(1, 2, 0).cpu().numpy()
    image_np = image_np * np.array([0.229, 0.224, 0.225], dtype=np.float32) + np.array([0.485, 0.456, 0.406], dtype=np.float32)
    image_np = np.clip(image_np, 0.0, 1.0)
    mask_np = mask.squeeze(0).cpu().numpy()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_np)
    axes[0].set_title(f"{sample_name} image")
    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title("ground truth")
    axes[2].imshow(prediction, cmap="gray")
    axes[2].set_title("prediction")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return str(save_path)

def save_history_figure(history, save_path):
    epochs = [item["epoch"] for item in history]
    train_dice = [item["train_dice"] for item in history]
    val_dice = [item["val_dice"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_dice, label="train dice")
    axes[0].plot(epochs, val_dice, label="val dice")
    axes[0].set_title("Dice")
    axes[0].legend()
    axes[1].plot(epochs, train_loss, label="train loss")
    axes[1].plot(epochs, val_loss, label="val loss")
    axes[1].set_title("Loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return str(save_path)
"""


EMCAD_BLOCK = """
assert torch is not None, "需要先安装 PyTorch 才能运行本 notebook。"

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x, h, w):
        b, n, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, h, w):
        x = self.fc1(x)
        x = self.dwconv(x, h, w)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=1, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x, h, w):
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        source = x
        if self.sr is not None:
            source = source.transpose(1, 2).reshape(b, c, h, w)
            source = self.sr(source).reshape(b, c, -1).transpose(1, 2)
            source = self.norm(source)
        kv = self.kv(source).reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)

class PVTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x, h, w):
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.mlp(self.norm2(x), h, w)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, kernel_size, stride, padding):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w

class PVTv2B0Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed1 = PatchEmbed(3, 32, kernel_size=7, stride=4, padding=3)
        self.patch_embed2 = PatchEmbed(32, 64, kernel_size=3, stride=2, padding=1)
        self.patch_embed3 = PatchEmbed(64, 160, kernel_size=3, stride=2, padding=1)
        self.patch_embed4 = PatchEmbed(160, 256, kernel_size=3, stride=2, padding=1)
        self.block1 = nn.ModuleList([PVTBlock(32, num_heads=1, mlp_ratio=8, sr_ratio=8) for _ in range(2)])
        self.block2 = nn.ModuleList([PVTBlock(64, num_heads=2, mlp_ratio=8, sr_ratio=4) for _ in range(2)])
        self.block3 = nn.ModuleList([PVTBlock(160, num_heads=5, sr_ratio=2) for _ in range(2)])
        self.block4 = nn.ModuleList([PVTBlock(256, num_heads=8, sr_ratio=1) for _ in range(2)])
        self.norm1 = nn.LayerNorm(32)
        self.norm2 = nn.LayerNorm(64)
        self.norm3 = nn.LayerNorm(160)
        self.norm4 = nn.LayerNorm(256)

    def forward(self, x):
        x1, h1, w1 = self.patch_embed1(x)
        for block in self.block1:
            x1 = block(x1, h1, w1)
        x1 = self.norm1(x1)
        x1_map = x1.transpose(1, 2).reshape(x1.shape[0], 32, h1, w1)

        x2, h2, w2 = self.patch_embed2(x1_map)
        for block in self.block2:
            x2 = block(x2, h2, w2)
        x2 = self.norm2(x2)
        x2_map = x2.transpose(1, 2).reshape(x2.shape[0], 64, h2, w2)

        x3, h3, w3 = self.patch_embed3(x2_map)
        for block in self.block3:
            x3 = block(x3, h3, w3)
        x3 = self.norm3(x3)
        x3_map = x3.transpose(1, 2).reshape(x3.shape[0], 160, h3, w3)

        x4, h4, w4 = self.patch_embed4(x3_map)
        for block in self.block4:
            x4 = block(x4, h4, w4)
        x4 = self.norm4(x4)
        x4_map = x4.transpose(1, 2).reshape(x4.shape[0], 256, h4, w4)
        return x1_map, x2_map, x3_map, x4_map

    def load_pretrained(self, weight_path):
        assert weight_path.exists(), f"未找到预训练权重: {weight_path}"
        state = torch.load(weight_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = self.load_state_dict(state, strict=False)
        print({"loaded_weight_path": str(weight_path), "missing_keys": len(missing), "unexpected_keys": len(unexpected)})

class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)

class MSCAM(nn.Module):
    def __init__(self, channels, kernel_sizes=(1, 3, 5)):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=k, padding=k // 2, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.GELU(),
                )
                for k in kernel_sizes
            ]
        )
        self.proj = nn.Conv2d(channels * len(kernel_sizes), channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.proj(torch.cat([branch(x) for branch in self.branches], dim=1))

class CAB(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x):
        weights = self.pool(x)
        weights = F.gelu(self.fc1(weights))
        weights = torch.sigmoid(self.fc2(weights))
        return x * weights

class SAB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(self, x):
        return x * torch.sigmoid(self.conv(x))

class EMCADDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, kernel_sizes=(1, 3, 5)):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.fuse = ConvNormAct(in_channels + skip_channels, out_channels, kernel_size=1)
        self.ms_cam = MSCAM(out_channels, kernel_sizes=kernel_sizes)
        self.cab = CAB(out_channels)
        self.sab = SAB(out_channels)
        self.refine = ConvNormAct(out_channels, out_channels, kernel_size=3)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.ms_cam(x)
        x = self.cab(x)
        x = self.sab(x)
        return self.refine(x)

class EMCADBaseline(nn.Module):
    def __init__(self, kernel_sizes=(1, 3, 5)):
        super().__init__()
        self.encoder = PVTv2B0Encoder()
        self.dec3 = EMCADDecoderBlock(256, 160, 160, kernel_sizes=kernel_sizes)
        self.dec2 = EMCADDecoderBlock(160, 64, 64, kernel_sizes=kernel_sizes)
        self.dec1 = EMCADDecoderBlock(64, 32, 32, kernel_sizes=kernel_sizes)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def load_pretrained_backbone(self, weight_path):
        self.encoder.load_pretrained(weight_path)

    def forward(self, x):
        skip1, skip2, skip3, bottleneck = self.encoder(x)
        x = self.dec3(bottleneck, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        logits = self.head(x)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return F.interpolate(logits, scale_factor=4, mode="bilinear", align_corners=False)
"""


UNET_BLOCK = """
assert torch is not None, "需要先安装 PyTorch 才能运行本 notebook。"

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.block(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNetBaseline(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, base_channels=32, bilinear=True):
        super().__init__()
        factor = 2 if bilinear else 1
        c1, c2, c3, c4, c5 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        )
        self.inc = DoubleConv(n_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c5 // factor)
        self.up1 = Up(c5, c4 // factor, bilinear=bilinear)
        self.up2 = Up(c4, c3 // factor, bilinear=bilinear)
        self.up3 = Up(c3, c2 // factor, bilinear=bilinear)
        self.up4 = Up(c2, c1, bilinear=bilinear)
        self.head = OutConv(c1, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.head(x)
"""

SWIN_UNET_BLOCK = """
assert torch is not None, "需要先安装 PyTorch 才能运行本 notebook。"

class SwinMLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)

def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size * window_size, c)

def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / ((h // window_size) * (w // window_size)))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(b, h, w, -1)

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=7, num_heads=3, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        relative_size = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(relative_size, num_heads))
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flat = coords.flatten(1)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(b_ // num_windows, num_windows, self.num_heads, n, n)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        return self.proj_drop(x)

class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size if min(input_resolution) > window_size else 0
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwinMLP(dim=dim, mlp_ratio=mlp_ratio, drop=drop)
        if self.shift_size > 0:
            self.register_buffer("attn_mask", self._build_mask(input_resolution))
        else:
            self.attn_mask = None

    def _build_mask(self, input_resolution):
        h, w = input_resolution
        padded_h = int(math.ceil(h / self.window_size)) * self.window_size
        padded_w = int(math.ceil(w / self.window_size)) * self.window_size
        mask = torch.zeros((1, padded_h, padded_w, 1))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        count = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                mask[:, h_slice, w_slice, :] = count
                count += 1
        mask_windows = window_partition(mask, self.window_size).view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x):
        h, w = self.input_resolution
        b, l, c = x.shape
        assert l == h * w, "token length and resolution do not match"
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h)).permute(0, 2, 3, 1)
        padded_h, padded_w = x.shape[1], x.shape[2]
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        shifted_x = window_reverse(attn_windows, self.window_size, padded_h, padded_w)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x[:, :h, :w, :].contiguous().view(b, h * w, c)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=352, patch_size=4, in_channels=3, embed_dim=48):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        h, w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.norm = nn.LayerNorm(dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)

    def forward(self, x):
        h, w = self.input_resolution
        b, l, c = x.shape
        assert l == h * w, "token length and resolution do not match"
        x = x.view(b, h, w, c)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(b, -1, 4 * c)
        x = self.norm(x)
        return self.reduction(x)

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, dim * 2, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x):
        h, w = self.input_resolution
        b, l, c = x.shape
        assert l == h * w, "token length and resolution do not match"
        x = self.expand(x)
        x = x.view(b, h, w, 2, 2, c // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h * 2 * w * 2, c // 2)
        return self.norm(x)

class FinalPatchExpandX4(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, dim * 16, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = self.input_resolution
        b, l, c = x.shape
        assert l == h * w, "token length and resolution do not match"
        x = self.expand(x)
        x = x.view(b, h, w, 4, 4, c)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h * 4, w * 4, c)
        x = x.view(b, -1, c)
        return self.norm(x)

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size=7, mlp_ratio=4.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if idx % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                )
                for idx in range(depth)
            ]
        )
        self.input_resolution = input_resolution

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class SkipFusion(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x, skip):
        x = torch.cat([x, skip], dim=-1)
        x = self.norm(x)
        return self.proj(x)

class SwinUNetBaseline(nn.Module):
    def __init__(
        self,
        img_size=352,
        patch_size=4,
        in_channels=3,
        num_classes=1,
        embed_dim=48,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        resolution = img_size // patch_size
        dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
        self.encoder_layers = nn.ModuleList(
            [
                BasicLayer(dims[0], (resolution, resolution), depths[0], num_heads[0], window_size=window_size),
                BasicLayer(dims[1], (resolution // 2, resolution // 2), depths[1], num_heads[1], window_size=window_size),
                BasicLayer(dims[2], (resolution // 4, resolution // 4), depths[2], num_heads[2], window_size=window_size),
                BasicLayer(dims[3], (resolution // 8, resolution // 8), depths[3], num_heads[3], window_size=window_size),
            ]
        )
        self.downsamples = nn.ModuleList(
            [
                PatchMerging((resolution, resolution), dims[0]),
                PatchMerging((resolution // 2, resolution // 2), dims[1]),
                PatchMerging((resolution // 4, resolution // 4), dims[2]),
            ]
        )
        self.up3 = PatchExpand((resolution // 8, resolution // 8), dims[3])
        self.up2 = PatchExpand((resolution // 4, resolution // 4), dims[2])
        self.up1 = PatchExpand((resolution // 2, resolution // 2), dims[1])
        self.fuse3 = SkipFusion(dims[2] * 2, dims[2])
        self.fuse2 = SkipFusion(dims[1] * 2, dims[1])
        self.fuse1 = SkipFusion(dims[0] * 2, dims[0])
        self.decoder3 = BasicLayer(dims[2], (resolution // 4, resolution // 4), 2, num_heads[2], window_size=window_size)
        self.decoder2 = BasicLayer(dims[1], (resolution // 2, resolution // 2), 2, num_heads[1], window_size=window_size)
        self.decoder1 = BasicLayer(dims[0], (resolution, resolution), 2, num_heads[0], window_size=window_size)
        self.final_expand = FinalPatchExpandX4((resolution, resolution), dims[0])
        self.head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        x, h, w = self.patch_embed(x)
        skip1 = self.encoder_layers[0](x)
        x = self.downsamples[0](skip1)
        skip2 = self.encoder_layers[1](x)
        x = self.downsamples[1](skip2)
        skip3 = self.encoder_layers[2](x)
        x = self.downsamples[2](skip3)
        x = self.encoder_layers[3](x)

        x = self.up3(x)
        x = self.fuse3(x, skip3)
        x = self.decoder3(x)
        x = self.up2(x)
        x = self.fuse2(x, skip2)
        x = self.decoder2(x)
        x = self.up1(x)
        x = self.fuse1(x, skip1)
        x = self.decoder1(x)
        x = self.final_expand(x)
        b = x.shape[0]
        x = x.view(b, h * 4, w * 4, self.head.in_channels).permute(0, 3, 1, 2).contiguous()
        return self.head(x)
"""


EXEC_IMPORT_BASELINE = """
import json
from pathlib import Path

baseline_nb = json.loads(Path("01_emcad_full_training.ipynb").read_text(encoding="utf-8"))
for idx in [2, 4, 6, 8]:
    exec("".join(baseline_nb["cells"][idx]["source"]), globals())
"""


ABLATION_BLOCK = """
class EMCADAblationModel(EMCADBaseline):
    def __init__(self):
        super().__init__(kernel_sizes=(3,))
"""


IMPROVEMENT_BLOCK = """
class LearnableFusionHead(nn.Module):
    def __init__(self, in_channels=32, branches=3):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, 1, kernel_size=1) for _ in range(branches)])
        self.weights = nn.Parameter(torch.ones(branches))

    def forward(self, x):
        branch_logits = [layer(x) for layer in self.projections]
        fusion_weights = torch.softmax(self.weights, dim=0)
        return sum(weight * logit for weight, logit in zip(fusion_weights, branch_logits))

class ImprovedEMCADModel(nn.Module):
    def __init__(self, kernel_sizes=(1, 3, 5)):
        super().__init__()
        self.encoder = PVTv2B0Encoder()
        self.dec3 = EMCADDecoderBlock(256, 160, 160, kernel_sizes=kernel_sizes)
        self.dec2 = EMCADDecoderBlock(160, 64, 64, kernel_sizes=kernel_sizes)
        self.dec1 = EMCADDecoderBlock(64, 32, 32, kernel_sizes=kernel_sizes)
        self.head = LearnableFusionHead(in_channels=32, branches=3)

    def load_pretrained_backbone(self, weight_path):
        self.encoder.load_pretrained(weight_path)

    def forward(self, x):
        skip1, skip2, skip3, bottleneck = self.encoder(x)
        x = self.dec3(bottleneck, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        logits = self.head(x)
        return F.interpolate(logits, scale_factor=4, mode="bilinear", align_corners=False)
"""


def build_00():
    return notebook(
        [
            md_cell(
                """
# Experiment: Project Bootstrap And ETIS Data Inspection

本 notebook 只负责 ETIS 项目的准备与检查：

- 固定项目目标和运行顺序
- 检查你已经放好的 ETIS 目录和列表文件
- 确认 `train / val / test = 156 / 20 / 20`
- 固定后续所有模型共用的测试可视化样本
"""
            ),
            md_cell(
                """
## 加载基础工具

这里仅加载最基础的环境、目录和 JSON 工具，不包含模型与训练逻辑。
"""
            ),
            code_cell(COMMON_IMPORT),
            md_cell(
                """
## 保存统一项目配置

这一段把 ETIS 数据路径、预训练权重路径、训练超参数和统一输出目录写入配置文件，供后续所有 notebook 共用。
"""
            ),
            code_cell(CONFIG_BLOCK),
            md_cell(
                """
## 定义 ETIS 数据检查函数

这里直接针对你当前的 ETIS 目录结构进行检查：

- `train/images` 与 `train/masks`
- `val/images` 与 `val/masks`
- `test/images` 与 `test/masks`
- 三个 list 文件
"""
            ),
            code_cell(ETIS_BOOTSTRAP_BLOCK),
            md_cell(
                """
## 执行 ETIS 数据检查

这一段会验证每个 split 的图片数、mask 数、列表数是否一致，并确定后续统一可视化的测试样本。
"""
            ),
            code_cell(
                """
split_summaries = {split: collect_split_summary(split) for split in ["train", "val", "test"]}
fixed_visual_sample = infer_fixed_visual_sample()
PROJECT_CONFIG["fixed_visual_sample"] = fixed_visual_sample
save_json(PROJECT_CONFIG, ARTIFACTS / "project_config.json")
split_summaries, fixed_visual_sample
"""
            ),
            md_cell(
                """
## 可视化一个真实样本

这里展示固定测试样本的原图和标注，方便后续所有模型做统一对比。
"""
            ),
            code_cell(
                """
fig = visualize_pair("test", fixed_visual_sample)
fig
"""
            ),
            md_cell(
                """
## 保存 bootstrap 检查结果

这里把 ETIS 当前划分和统一可视化样本写入 records，作为后续实验的基础记录。
"""
            ),
            code_cell(
                """
save_json(
    {
        "env_summary": ENV_SUMMARY,
        "dataset": "ETIS",
        "split_summaries": split_summaries,
        "fixed_visual_sample": fixed_visual_sample,
        "pretrained_path": PROJECT_CONFIG["pvt_pretrained_path"],
    },
    ARTIFACTS / "records" / "bootstrap_etis_summary.json",
)
"""
            ),
        ]
    )


def build_01():
    return notebook(
        [
            md_cell(
                """
# Experiment: EMCAD Full Training On ETIS

这一册是 EMCAD baseline 的唯一完整定义来源，并且已经改为真正使用你放在 `data/ETIS/` 下的 polyp segmentation 数据。
"""
            ),
            md_cell(
                """
## 加载基础工具

这里只加载最基础的环境和结果保存工具，真实数据流、模型和训练逻辑都在本 notebook 内定义。
"""
            ),
            code_cell(COMMON_IMPORT),
            md_cell(
                """
## 读取共享实验配置

这一段直接读取 `00` 已写入的统一项目配置，因此后面的 notebook 不再重复维护同一份 `PROJECT_CONFIG`。
"""
            ),
            code_cell(CONFIG_LOAD_BLOCK),
            md_cell(
                """
## 定义 ETIS 数据管线与训练评估工具

下面的代码块提供 ETIS 的数据读取、Dice 评估、训练循环、测试评估和可视化导出函数。
"""
            ),
            code_cell(ETIS_SHARED_PIPELINE),
            md_cell(
                """
## 定义 EMCAD baseline 结构

下面的代码块给出本项目中 EMCAD baseline 的唯一完整实现来源，并显式支持加载 `pvt_v2_b0.pth`。
"""
            ),
            code_cell(EMCAD_BLOCK),
            md_cell(
                """
## 构建 ETIS dataloader 并检查预训练权重

这里会读取 `train / val / test` 三段数据，并检查 `data/pvt_pretrained_pth/pvt_v2_b0.pth` 是否存在。
"""
            ),
            code_cell(
                """
device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = PROJECT_CONFIG["train"]
train_loader, val_loader, test_loader = build_dataloaders(cfg)
weight_path = Path(PROJECT_CONFIG["pvt_pretrained_path"])
assert weight_path.exists(), f"未找到预训练权重: {weight_path}"
{
    "device": device,
    "train_count": len(train_loader.dataset),
    "val_count": len(val_loader.dataset),
    "test_count": len(test_loader.dataset),
    "fixed_visual_sample": default_visual_sample(),
    "pretrained_path": str(weight_path),
}
"""
            ),
            md_cell(
                """
## 实例化 EMCAD baseline 并加载 B0 预训练权重

这里显式加载你已准备好的 PVTv2-B0 权重，然后把 EMCAD 送到当前可用设备。
"""
            ),
            code_cell(
                """
model = EMCADBaseline().to(device)
model.load_pretrained_backbone(weight_path)
"""
            ),
            md_cell(
                """
## 运行完整训练与验证流程

这里按统一配置完成完整训练，并保留验证集上表现最好的模型权重。
"""
            ),
            code_cell(
                """
history, best_val_dice = train_model(model, train_loader, val_loader, cfg, device=device)
best_val_dice
"""
            ),
            md_cell(
                """
## 在验证集和测试集上评估 EMCAD

这里统一输出验证集摘要、测试集摘要和逐样本 Dice，方便后续对照、消融和改进读取。
"""
            ),
            code_cell(
                """
val_summary, val_rows = evaluate_loader(model, val_loader, device=device)
test_summary, test_rows = evaluate_loader(model, test_loader, device=device)
val_summary, test_summary
"""
            ),
            md_cell(
                """
## 导出训练曲线和统一测试样本可视化

这里把 EMCAD 的训练曲线和固定测试样本预测图导出到 artifacts，后续所有比较都引用这些产物。
"""
            ),
            code_cell(
                """
history_figure_path = ARTIFACTS / "figures" / "emcad_training_history.png"
saved_history_figure = save_history_figure(history, history_figure_path)
visual_sample = default_visual_sample()
visual_path = ARTIFACTS / "figures" / "emcad_visual_sample.png"
saved_visual = export_visualization(model, visual_sample, cfg["image_size"], visual_path, device=device)
saved_history_figure, saved_visual
"""
            ),
            md_cell(
                """
## 保存 EMCAD baseline 结果

这里输出的是后续所有对照、消融和改进实验要引用的 baseline 结果来源。
"""
            ),
            code_cell(
                """
checkpoint_path = ARTIFACTS / "checkpoints" / "emcad_b0_etis_best.pth"
torch.save(model.state_dict(), checkpoint_path)

save_json(
    {
        "dataset": "ETIS",
        "emcad_scale": PROJECT_CONFIG["emcad_scale"],
        "device": device,
        "pretrained_path": str(weight_path),
        "history": history,
        "best_val_dice": best_val_dice,
        "val_summary": val_summary,
        "test_summary": test_summary,
        "val_rows": val_rows,
        "test_rows": test_rows,
        "visual_sample": visual_sample,
        "visual_path": saved_visual,
        "history_figure_path": saved_history_figure,
        "checkpoint_path": str(checkpoint_path),
    },
    ARTIFACTS / "records" / "emcad_etis_results.json",
)
"""
            ),
        ]
    )


def build_02():
    return notebook(
        [
            md_cell(
                """
# Experiment: U-Net And Swin-Unet Comparison On ETIS

这一册完整实现 U-Net 和 Swin-Unet，并与 `01` 中保存的 EMCAD baseline 结果做统一口径比较。
"""
            ),
            md_cell(
                """
## 加载基础工具

这里只使用环境与结果保存工具，所有 ETIS 数据读取和模型定义都写在本 notebook 内。
"""
            ),
            code_cell(COMMON_IMPORT),
            md_cell(
                """
## 读取共享实验配置

这里直接复用 `00` 生成的统一配置，保证和 `01` 的 ETIS 路径、训练参数与固定可视化样本保持一致。
"""
            ),
            code_cell(CONFIG_LOAD_BLOCK),
            md_cell(
                """
## 复用 ETIS 数据与评估工具

这里直接复用 `01` 的 ETIS 数据流与 Dice 评估逻辑，确保对照实验和 EMCAD baseline 完全同口径。
"""
            ),
            code_cell(EXEC_IMPORT_BASELINE),
            md_cell(
                """
## 定义 U-Net

这一段采用接近官方 `Pytorch-UNet` 的经典编码器 - 解码器结构，保留四次下采样、四次上采样和 skip connection。
"""
            ),
            code_cell(UNET_BLOCK),
            md_cell(
                """
## 定义 Swin-Unet

这一段参考官方 `Swin-Unet` 的核心思路，保留 patch embedding、分层 Swin Transformer block、patch merging、patch expand 和 U 形跳连解码结构，而不是只用简单卷积替代。
"""
            ),
            code_cell(SWIN_UNET_BLOCK),
            md_cell(
                """
## 读取 EMCAD baseline 结果

本 notebook 不再重复定义 EMCAD，只读取 `01` 里保存的基准实验结果。
"""
            ),
            code_cell(
                """
emcad_result_path = ARTIFACTS / "records" / "emcad_etis_results.json"
emcad_reference = json.loads(emcad_result_path.read_text(encoding="utf-8")) if emcad_result_path.exists() else {}
emcad_reference.get("test_summary", {})
"""
            ),
            md_cell(
                """
## 构建 ETIS dataloader

这里继续沿用统一的 `train / val / test = 156 / 20 / 20`。
"""
            ),
            code_cell(
                """
device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = PROJECT_CONFIG["train"]
train_loader, val_loader, test_loader = build_dataloaders(cfg)
{
    "device": device,
    "train_count": len(train_loader.dataset),
    "val_count": len(val_loader.dataset),
    "test_count": len(test_loader.dataset),
    "visual_sample": default_visual_sample(),
}
"""
            ),
            md_cell(
                """
## 训练 U-Net 和 Swin-Unet 并在 ETIS 上评估

这里为两个模型执行统一流程，确保与 EMCAD baseline 的比较具有同一数据口径。
"""
            ),
            code_cell(
                """
comparison_rows = []
model_specs = {
    "U-Net": UNetBaseline(base_channels=32),
    "Swin-Unet": SwinUNetBaseline(embed_dim=48),
}

for model_name, model in model_specs.items():
    model = model.to(device)
    history, best_val_dice = train_model(model, train_loader, val_loader, cfg, device=device)
    val_summary, val_rows = evaluate_loader(model, val_loader, device=device)
    test_summary, test_rows = evaluate_loader(model, test_loader, device=device)
    history_path = ARTIFACTS / "figures" / f"{model_name.lower().replace('-', '_')}_history.png"
    visual_path = ARTIFACTS / "figures" / f"{model_name.lower().replace('-', '_')}_visual_sample.png"
    saved_history_path = save_history_figure(history, history_path)
    saved_visual_path = export_visualization(model, default_visual_sample(), cfg["image_size"], visual_path, device=device)
    checkpoint_path = ARTIFACTS / "checkpoints" / f"{model_name.lower().replace('-', '_')}_etis_best.pth"
    torch.save(model.state_dict(), checkpoint_path)
    comparison_rows.append(
        {
            "model": model_name,
            "best_val_dice": best_val_dice,
            "val_summary": val_summary,
            "test_summary": test_summary,
            "val_rows": val_rows,
            "test_rows": test_rows,
            "history_figure_path": saved_history_path,
            "visual_path": saved_visual_path,
            "checkpoint_path": str(checkpoint_path),
        }
    )

if emcad_reference:
    comparison_rows.append(
        {
            "model": "EMCAD",
            "best_val_dice": emcad_reference["best_val_dice"],
            "val_summary": emcad_reference["val_summary"],
            "test_summary": emcad_reference["test_summary"],
            "val_rows": emcad_reference["val_rows"],
            "test_rows": emcad_reference["test_rows"],
            "history_figure_path": emcad_reference["history_figure_path"],
            "visual_path": emcad_reference["visual_path"],
            "checkpoint_path": emcad_reference["checkpoint_path"],
        }
    )

comparison_rows
"""
            ),
            md_cell(
                """
## 保存 ETIS 对照结果

输出表中会包含每个模型对同一个测试样本导出的分割图路径，便于后续肉眼直接比较。
"""
            ),
            code_cell(
                """
save_json(
    {
        "dataset": "ETIS",
        "device": device,
        "visual_sample": default_visual_sample(),
        "rows": comparison_rows,
    },
    ARTIFACTS / "records" / "baseline_comparison_etis.json",
)
"""
            ),
        ]
    )


def build_03():
    return notebook(
        [
            md_cell(
                """
# Experiment: EMCAD Ablation And Failure Analysis On ETIS

这一册基于真实 ETIS 数据运行消融版本，并与 `01` 的 baseline 结果做统一对比。
"""
            ),
            md_cell(
                """
## 加载基础工具

这里只使用环境与结果保存工具，真实消融逻辑都写在本 notebook 内。
"""
            ),
            code_cell(COMMON_IMPORT),
            md_cell(
                """
## 读取共享实验配置

这里直接读取 `00` 已保存的统一配置，继续沿用 `01` 的 ETIS 路径、训练参数和统一测试样本。
"""
            ),
            code_cell(CONFIG_LOAD_BLOCK),
            md_cell(
                """
## 读取 EMCAD baseline 结果

baseline 结果固定来自 `01_emcad_full_training.ipynb`。
"""
            ),
            code_cell(
                """
baseline_result_path = ARTIFACTS / "records" / "emcad_etis_results.json"
baseline_reference = json.loads(baseline_result_path.read_text(encoding="utf-8")) if baseline_result_path.exists() else {}
baseline_reference.get("test_summary", {})
"""
            ),
            md_cell(
                """
## 复用 baseline 数据与组件

这里直接复用 `01` 的 ETIS 数据流和 EMCAD 结构组件，但不再重复整段 baseline notebook 内容。
"""
            ),
            code_cell(EXEC_IMPORT_BASELINE),
            md_cell(
                """
## 定义消融模型

这里不复制 baseline 全结构，只定义消融后发生变化的地方：多尺度卷积核由 `(1, 3, 5)` 变为 `(3,)`。
"""
            ),
            code_cell(ABLATION_BLOCK),
            md_cell(
                """
## 说明 baseline 与 ablation 的结构差异

这一段明确指出消融改动点及其分析目标。
"""
            ),
            code_cell(
                """
ablation_design = {
    "baseline_source": "01_emcad_full_training.ipynb",
    "baseline_summary": "EMCAD baseline uses MSCAM with kernel sizes (1, 3, 5).",
    "ablation_change": "Replace multi-scale MSCAM kernels with a single scale kernel set (3,).",
    "analysis_goal": "Measure how much explicit multi-scale decoding contributes on the ETIS split.",
}
ablation_design
"""
            ),
            md_cell(
                """
## 在 ETIS 上训练消融模型

这里继续使用同一数据和同一训练流程，对消融版本完成完整训练与测试。
"""
            ),
            code_cell(
                """
device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = PROJECT_CONFIG["train"]
train_loader, val_loader, test_loader = build_dataloaders(cfg)
weight_path = Path(PROJECT_CONFIG["pvt_pretrained_path"])
assert weight_path.exists(), f"未找到预训练权重: {weight_path}"

ablation_model = EMCADAblationModel().to(device)
ablation_model.load_pretrained_backbone(weight_path)
ablation_history, ablation_best_val_dice = train_model(ablation_model, train_loader, val_loader, cfg, device=device)
ablation_val_summary, ablation_val_rows = evaluate_loader(ablation_model, val_loader, device=device)
ablation_test_summary, ablation_test_rows = evaluate_loader(ablation_model, test_loader, device=device)
ablation_test_summary
"""
            ),
            md_cell(
                """
## 导出消融模型的统一测试样本结果

这里继续使用与 baseline 相同的测试样本，方便直接观察多尺度模块简化后带来的差异。
"""
            ),
            code_cell(
                """
ablation_history_path = ARTIFACTS / "figures" / "emcad_ablation_history.png"
saved_ablation_history = save_history_figure(ablation_history, ablation_history_path)
ablation_visual_path = ARTIFACTS / "figures" / "emcad_ablation_visual_sample.png"
saved_ablation_visual = export_visualization(
    ablation_model,
    default_visual_sample(),
    cfg["image_size"],
    ablation_visual_path,
    device=device,
)
saved_ablation_visual
"""
            ),
            md_cell(
                """
## 生成 baseline 与 ablation 对比

这里把 `01` 的 baseline 结果和当前消融结果放在同一张表里，突出结构差异带来的影响。
"""
            ),
            code_cell(
                """
comparison = []
if baseline_reference:
    comparison.append(
        {
            "variant": "EMCAD baseline",
            "best_val_dice": baseline_reference["best_val_dice"],
            "test_summary": baseline_reference["test_summary"],
            "difference": "Reference from 01 with MSCAM kernels (1, 3, 5)",
            "visual_path": baseline_reference["visual_path"],
        }
    )

comparison.append(
    {
        "variant": "Single-scale MSCAM ablation",
        "best_val_dice": ablation_best_val_dice,
        "test_summary": ablation_test_summary,
        "difference": "MSCAM kernels changed to (3,)",
        "visual_path": saved_ablation_visual,
    }
)
comparison
"""
            ),
            md_cell(
                """
## 写入失败分析入口

这里把 ETIS 环境下的失败分析入口固定下来，后续可直接结合同一测试样本的分割图做肉眼分析。
"""
            ),
            code_cell(
                """
ablation_checkpoint_path = ARTIFACTS / "checkpoints" / "emcad_ablation_etis_best.pth"
torch.save(ablation_model.state_dict(), ablation_checkpoint_path)

failure_analysis = {
    "baseline_source": "01_emcad_full_training.ipynb",
    "visual_sample": default_visual_sample(),
    "baseline_visual_path": baseline_reference.get("visual_path"),
    "ablation_visual_path": saved_ablation_visual,
    "failure_modes": [
        {
            "name": "small polyp under-segmentation",
            "hypothesis": "single-scale decoding weakens multi-scale context aggregation",
            "evidence_placeholder": "attach per-sample Dice and qualitative figures here",
        },
        {
            "name": "boundary leakage",
            "hypothesis": "reduced receptive field hurts ambiguous polyp boundaries",
            "evidence_placeholder": "attach boundary comparison figures here",
        },
    ],
}

save_json(
    {
        "dataset": "ETIS",
        "pretrained_path": str(weight_path),
        "ablation_design": ablation_design,
        "history": ablation_history,
        "best_val_dice": ablation_best_val_dice,
        "val_summary": ablation_val_summary,
        "test_summary": ablation_test_summary,
        "val_rows": ablation_val_rows,
        "test_rows": ablation_test_rows,
        "history_figure_path": saved_ablation_history,
        "visual_sample": default_visual_sample(),
        "visual_path": saved_ablation_visual,
        "checkpoint_path": str(ablation_checkpoint_path),
        "comparison": comparison,
        "failure_analysis": failure_analysis,
    },
    ARTIFACTS / "records" / "ablation_and_failure_analysis_etis.json",
)
"""
            ),
        ]
    )


def build_04():
    return notebook(
        [
            md_cell(
                """
# Experiment: EMCAD Improvement On ETIS

这一册基于真实 ETIS 数据，只围绕 prediction head 的轻量结构改进做对比，而不重复完整 baseline。
"""
            ),
            md_cell(
                """
## 加载基础工具

这里只保留环境与结果保存工具，真实改进实验逻辑都写在本 notebook 内。
"""
            ),
            code_cell(COMMON_IMPORT),
            md_cell(
                """
## 读取共享实验配置

这里直接读取 `00` 已保存的统一配置，继续沿用 `01` 的 ETIS 数据路径和统一训练参数。
"""
            ),
            code_cell(CONFIG_LOAD_BLOCK),
            md_cell(
                """
## 读取 EMCAD baseline 结果

baseline 结果固定来自 `01_emcad_full_training.ipynb`。
"""
            ),
            code_cell(
                """
baseline_result_path = ARTIFACTS / "records" / "emcad_etis_results.json"
baseline_reference = json.loads(baseline_result_path.read_text(encoding="utf-8")) if baseline_result_path.exists() else {}
baseline_reference.get("test_summary", {})
"""
            ),
            md_cell(
                """
## 复用 baseline 数据与 EMCAD 组件

这里继续复用 `01` 的 ETIS 数据流与 EMCAD 组件，只单独定义改进过的 prediction head。
"""
            ),
            code_cell(EXEC_IMPORT_BASELINE),
            md_cell(
                """
## 定义改进模块

这里不重写整个 baseline，只定义 prediction head 的差异：

- baseline：单头预测
- improvement：可学习融合头
"""
            ),
            code_cell(IMPROVEMENT_BLOCK),
            md_cell(
                """
## 说明 baseline 与改进版的结构差异

这一段明确改进前后变化的范围和预期收益。
"""
            ),
            code_cell(
                """
improvement_design = {
    "baseline_source": "01_emcad_full_training.ipynb",
    "baseline_summary": "EMCAD baseline uses a single 1x1 prediction head after decoder refinement.",
    "improvement_change": "Replace the single prediction head with a learnable multi-branch fusion head.",
    "expected_benefit": "Improve final decoder output aggregation without changing the whole backbone and decoder stack.",
}
improvement_design
"""
            ),
            md_cell(
                """
## 在 ETIS 上训练改进版 EMCAD

这里继续使用同一套 ETIS 训练、验证和测试流程，对改进版完成完整训练与评估。
"""
            ),
            code_cell(
                """
device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = PROJECT_CONFIG["train"]
train_loader, val_loader, test_loader = build_dataloaders(cfg)
weight_path = Path(PROJECT_CONFIG["pvt_pretrained_path"])
assert weight_path.exists(), f"未找到预训练权重: {weight_path}"

improved_model = ImprovedEMCADModel().to(device)
improved_model.load_pretrained_backbone(weight_path)
improvement_history, improvement_best_val_dice = train_model(improved_model, train_loader, val_loader, cfg, device=device)
improvement_val_summary, improvement_val_rows = evaluate_loader(improved_model, val_loader, device=device)
improvement_test_summary, improvement_test_rows = evaluate_loader(improved_model, test_loader, device=device)
improvement_test_summary
"""
            ),
            md_cell(
                """
## 导出改进版的统一测试样本结果

这里固定导出与 baseline 相同的测试样本，便于直接比较改进前后的预测差异。
"""
            ),
            code_cell(
                """
improvement_history_path = ARTIFACTS / "figures" / "emcad_improved_history.png"
saved_improvement_history = save_history_figure(improvement_history, improvement_history_path)
improvement_visual_path = ARTIFACTS / "figures" / "emcad_improved_visual_sample.png"
saved_improvement_visual = export_visualization(
    improved_model,
    default_visual_sample(),
    cfg["image_size"],
    improvement_visual_path,
    device=device,
)
saved_improvement_visual
"""
            ),
            md_cell(
                """
## 生成 baseline 与改进版对比

这里把 `01` 的 baseline 结果与当前改进版结果放到一起，突出改动点和结果解释。
"""
            ),
            code_cell(
                """
comparison = []
if baseline_reference:
    comparison.append(
        {
            "variant": "EMCAD baseline",
            "best_val_dice": baseline_reference["best_val_dice"],
            "test_summary": baseline_reference["test_summary"],
            "difference": "Reference baseline result",
            "visual_path": baseline_reference["visual_path"],
        }
    )

comparison.append(
    {
        "variant": "Improved learnable fusion head",
        "best_val_dice": improvement_best_val_dice,
        "test_summary": improvement_test_summary,
        "difference": "Replace the single prediction head with a multi-branch learnable fusion head",
        "visual_path": saved_improvement_visual,
    }
)
comparison
"""
            ),
            md_cell(
                """
## 保存改进实验记录

这里会把改进设计、统一测试样本和对比结果一起写入 records。
"""
            ),
            code_cell(
                """
improvement_checkpoint_path = ARTIFACTS / "checkpoints" / "emcad_improved_etis_best.pth"
torch.save(improved_model.state_dict(), improvement_checkpoint_path)

save_json(
    {
        "dataset": "ETIS",
        "pretrained_path": str(weight_path),
        "improvement_design": improvement_design,
        "history": improvement_history,
        "best_val_dice": improvement_best_val_dice,
        "val_summary": improvement_val_summary,
        "test_summary": improvement_test_summary,
        "val_rows": improvement_val_rows,
        "test_rows": improvement_test_rows,
        "history_figure_path": saved_improvement_history,
        "visual_sample": default_visual_sample(),
        "visual_path": saved_improvement_visual,
        "checkpoint_path": str(improvement_checkpoint_path),
        "comparison": comparison,
    },
    ARTIFACTS / "records" / "improvement_experiment_etis.json",
)
"""
            ),
        ]
    )


def main():
    notebooks = {
        "00_project_bootstrap_etis.ipynb": build_00(),
        "01_emcad_full_training.ipynb": build_01(),
        "02_baseline_comparison.ipynb": build_02(),
        "03_ablation_and_failure_analysis.ipynb": build_03(),
        "04_improvement_experiment.ipynb": build_04(),
    }
    for path in ROOT.glob("*.ipynb"):
        path.unlink()
    for name, data in notebooks.items():
        (ROOT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {ROOT / name}")


if __name__ == "__main__":
    main()
