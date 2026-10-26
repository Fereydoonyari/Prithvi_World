"""Load frozen Prithvi-EO-2.0 (num_frames=1) and decode tokens to a patch."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from .paths import MODEL_DIR, SOFTWARE_ROOT, _ensure_dir

if str(SOFTWARE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOFTWARE_ROOT))


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def download_model(repo_id: str, local_dir: Path | None = None) -> Path:
    local_dir = Path(local_dir or MODEL_DIR)
    _ensure_dir(local_dir)
    ckpt = local_dir / "Prithvi_EO_V2_100M_TL.pt"
    cfg = local_dir / "config.json"
    if ckpt.exists() and cfg.exists():
        return local_dir
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    return local_dir


def load_prithvi_config(model_dir: Path | None = None) -> dict:
    model_dir = Path(model_dir or MODEL_DIR)
    with open(model_dir / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)["pretrained_cfg"]


def load_single_frame_prithvi(
    repo_id: str = "ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL",
    model_dir: Path | None = None,
    device: torch.device | None = None,
):
    device = device or get_device()
    model_dir = download_model(repo_id, model_dir)
    config = dict(load_prithvi_config(model_dir))
    config["num_frames"] = 1
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from prithvi_mae import PrithviMAE

    model = PrithviMAE(**config)
    ckpt_path = model_dir / "Prithvi_EO_V2_100M_TL.pt"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = dict(checkpoint)
    for key in list(state):
        if "pos_embed" in key:
            del state[key]
    missing, unexpected = model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "model": model,
        "config": config,
        "device": device,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "num_parameters": n_params,
        "model_dir": model_dir,
        "repo_id": repo_id,
        "checkpoint_path": ckpt_path,
    }


def denormalize_prithvi(normalized: "np.ndarray") -> "np.ndarray":
    import numpy as np

    from .hls import PRITHVI_MEAN, PRITHVI_STD

    x = np.asarray(normalized, dtype=np.float32)
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 3 or x.shape[0] != 6:
        raise ValueError(f"Expected (6,H,W), got {x.shape}")
    return x * PRITHVI_STD[:, None, None] + PRITHVI_MEAN[:, None, None]


@torch.no_grad()
def decode_tokens_to_patch(
    model,
    device: torch.device,
    tokens,
    cls,
    year: int,
    doy: int,
    lat: float,
    lon: float,
    img_size: int = 224,
):
    tok = torch.as_tensor(tokens, device=device, dtype=torch.float32)
    if tok.ndim != 3:
        raise ValueError(f"tokens must be (14,14,768), got {tuple(tok.shape)}")
    h, w, dim = tok.shape
    cls_t = torch.as_tensor(cls, device=device, dtype=torch.float32).reshape(1, 1, dim)
    latent = torch.cat([cls_t, tok.reshape(1, h * w, dim)], dim=1)
    ids_restore = torch.arange(h * w, device=device).unsqueeze(0)
    tcoords = torch.tensor([[[float(year), float(doy)]]], dtype=torch.float32, device=device)
    lcoords = torch.tensor([[float(lat), float(lon)]], dtype=torch.float32, device=device)
    pred = model.decoder(
        latent,
        ids_restore,
        temporal_coords=tcoords,
        location_coords=lcoords,
        input_size=(1, 6, 1, img_size, img_size),
    )
    patch = model.unpatchify(pred, image_size=(img_size, img_size))
    if patch.ndim == 5:
        patch = patch[:, :, 0]
    return patch[0].detach().cpu().numpy()
