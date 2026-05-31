from pathlib import Path
import json
import os
import platform
import random
import sys


ROOT = Path.cwd().resolve()
ARTIFACTS = ROOT / "artifacts"
DATA_ROOT = ROOT / "data"
ETIS_ROOT = DATA_ROOT / "ETIS"
PVT_PRETRAINED_ROOT = DATA_ROOT / "pvt_pretrained_pth"
SWIN_UNET_PRETRAINED_ROOT = DATA_ROOT / "SwinUnet_pretrained_pth"
SEED = 42


def ensure_project_dirs():
    for path in [
        DATA_ROOT,
        ETIS_ROOT,
        PVT_PRETRAINED_ROOT,
        SWIN_UNET_PRETRAINED_ROOT,
        ARTIFACTS,
        ARTIFACTS / "checkpoints",
        ARTIFACTS / "figures",
        ARTIFACTS / "records",
        ROOT / "md",
        ROOT / "scripts",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def try_import_torch():
    try:
        import torch

        return torch
    except Exception:
        return None


def get_default_project_config():
    return {
        "dataset": "ETIS",
        "task": "Polyp Segmentation",
        "paper_repo": "https://github.com/SLDGroup/EMCAD",
        "baseline_repos": {
            "Swin-Unet": "https://github.com/HuCaoFighting/Swin-Unet",
            "U-Net": "https://github.com/milesial/Pytorch-UNet",
        },
        "emcad_scale": "PVT-EMCAD-B0",
        "metrics": ["Dice"],
        "fixed_visual_sample": None,
        "etis_paths": {
            "root": str(ETIS_ROOT),
            "train_images": str(ETIS_ROOT / "train" / "images"),
            "train_masks": str(ETIS_ROOT / "train" / "masks"),
            "val_images": str(ETIS_ROOT / "val" / "images"),
            "val_masks": str(ETIS_ROOT / "val" / "masks"),
            "test_images": str(ETIS_ROOT / "test" / "images"),
            "test_masks": str(ETIS_ROOT / "test" / "masks"),
            "train_list": str(ETIS_ROOT / "train_list_etis.txt"),
            "val_list": str(ETIS_ROOT / "val_list_etis.txt"),
            "test_list": str(ETIS_ROOT / "test_list_etis.txt"),
        },
        "pvt_pretrained_path": str(PVT_PRETRAINED_ROOT / "pvt_v2_b0.pth"),
        "swin_unet_pretrained_path": str(SWIN_UNET_PRETRAINED_ROOT / "swin_tiny_patch4_window7_224.pth"),
        "train": {
            "epochs": 60,
            "batch_size": 8,
            "image_size": 352,
            "num_workers": 0,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
        },
    }


def _merge_with_defaults(default_value, current_value):
    if isinstance(default_value, dict) and isinstance(current_value, dict):
        merged = dict(default_value)
        for key, value in current_value.items():
            if key in merged:
                merged[key] = _merge_with_defaults(merged[key], value)
            else:
                merged[key] = value
        return merged
    return current_value if current_value is not None else default_value


def load_project_config(config_path=None):
    config_path = Path(config_path) if config_path is not None else ARTIFACTS / "project_config.json"
    default_config = get_default_project_config()
    if config_path.exists():
        loaded_config = json.loads(config_path.read_text(encoding="utf-8"))
        merged_config = _merge_with_defaults(default_config, loaded_config)
        if merged_config != loaded_config:
            save_json(merged_config, config_path)
        return merged_config
    config = default_config
    save_json(config, config_path)
    return config


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def print_env_summary(torch_module=None):
    summary = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "root": str(ROOT),
        "data_root": str(DATA_ROOT),
        "etis_root": str(ETIS_ROOT),
        "pvt_pretrained_root": str(PVT_PRETRAINED_ROOT),
        "swin_unet_pretrained_root": str(SWIN_UNET_PRETRAINED_ROOT),
        "torch_installed": torch_module is not None,
        "cuda_available": bool(torch_module and torch_module.cuda.is_available()),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
