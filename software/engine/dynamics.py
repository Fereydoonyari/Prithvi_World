"""Latent dynamics: residual ConvLSTM forecast of Prithvi token grids.

The network predicts Δz; ẑ_{t+1} = z_t + Δz. Prithvi stays frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .hls import date_id_to_datetime
from .similarity import cosine_similarity_map


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------


def _item_datetime(item: dict) -> datetime:
    meta = item.get("metadata") or {}
    date_id = meta.get("acquisition_date_id")
    if date_id:
        return date_id_to_datetime(str(date_id))
    key = item.get("key") or meta.get("date") or ""
    if len(key) >= 10 and key[4] == "-" and key[7] == "-":
        return datetime.fromisoformat(key[:10])
    if len(key) >= 7:
        return datetime.fromisoformat(key[:7] + "-01")
    raise ValueError(f"Cannot parse date from embedding item keys={key!r}")


def _doy_encoding(dt: datetime) -> np.ndarray:
    doy = float(dt.timetuple().tm_yday)
    ang = 2.0 * np.pi * doy / 365.25
    return np.array([np.sin(ang), np.cos(ang)], dtype=np.float32)


def _seasonal_copy_item(ordered: list[dict], target_item: dict) -> dict | None:
    """Same location, previous year, closest day-of-year. None if no prior year."""
    t = _item_datetime(target_item)
    want_year = t.year - 1
    want_doy = t.timetuple().tm_yday
    cands = [it for it in ordered if _item_datetime(it).year == want_year]
    if not cands:
        return None
    return min(
        cands,
        key=lambda it: abs(_item_datetime(it).timetuple().tm_yday - want_doy),
    )


def _circular_doy_gap(a: int, b: int) -> int:
    d = abs(a - b)
    return min(d, 365 - d)


def _climatology_items(
    ordered: list[dict],
    target_item: dict,
    window_days: int = 20,
    past_only: bool = True,
) -> list[dict]:
    """Observations from earlier years within ``window_days`` of the target DOY.

    A single previous-year scene (``_seasonal_copy_item``) is one noisy draw from
    the seasonal cycle. Averaging several years near the same day-of-year is the
    climatology baseline the vegetation-forecasting literature compares against,
    and it is markedly harder to beat.

    ``past_only`` keeps the baseline causal: a forecaster standing at the target
    date could not have seen later years. It does not change the 2024 test
    windows (whose only other years are 2022–2023) but it stops earlier windows
    from averaging in their own future.
    """
    t = _item_datetime(target_item)
    want_doy = t.timetuple().tm_yday
    out = []
    for it in ordered:
        year = _item_datetime(it).year
        if year == t.year:
            continue
        if past_only and year > t.year:
            continue
        if _circular_doy_gap(_item_datetime(it).timetuple().tm_yday, want_doy) <= window_days:
            out.append(it)
    return out


@dataclass
class ForecastSample:
    past: np.ndarray  # (K, 14, 14, 768)
    target: np.ndarray  # (14, 14, 768)
    seasonal: np.ndarray  # (14, 14, 768) same DOY previous year (or z_t if missing)
    delta_days: np.ndarray  # (K,) days from each past frame to the target
    doy_sincos: np.ndarray  # (2,) of the *target* date
    location_id: str
    past_keys: list[str]
    target_key: str
    has_seasonal: bool
    seasonal_key: str | None = None
    # Mean of all other-year scenes near the target day-of-year (or z_t if none).
    climatology: np.ndarray | None = None
    has_climatology: bool = False
    climatology_keys: list[str] | None = None


def build_forecast_samples(
    sequences: dict[str, list[dict]],
    history: int = 6,
    min_delta_days: float = 21.0,
    climatology_window_days: int = 20,
    climatology_past_only: bool = True,
) -> list[ForecastSample]:
    """Past K states → first future observation at least ``min_delta_days`` later.

    Skip 5-day next-frame steps; those are almost identity and make persistence trivial.
    """
    if history < 1:
        raise ValueError("history must be >= 1")
    samples: list[ForecastSample] = []
    for loc_id, seq in sequences.items():
        ordered = sorted(seq, key=_item_datetime)
        n = len(ordered)
        if n < history + 1:
            continue
        for last_idx in range(history - 1, n - 1):
            past_items = ordered[last_idx - history + 1 : last_idx + 1]
            t_last = _item_datetime(past_items[-1])
            target_item = None
            for k in range(last_idx + 1, n):
                dt_days = (_item_datetime(ordered[k]) - t_last).days
                if dt_days >= min_delta_days:
                    target_item = ordered[k]
                    break
            if target_item is None:
                continue
            t_target = _item_datetime(target_item)
            deltas = np.array(
                [(t_target - _item_datetime(p)).days for p in past_items],
                dtype=np.float32,
            )
            seasonal_item = _seasonal_copy_item(ordered, target_item)
            has_seasonal = seasonal_item is not None
            seasonal_src = seasonal_item if has_seasonal else past_items[-1]
            seasonal = seasonal_src["tokens"].astype(np.float32)

            clim_items = _climatology_items(
                ordered,
                target_item,
                window_days=climatology_window_days,
                past_only=climatology_past_only,
            )
            has_climatology = bool(clim_items)
            if has_climatology:
                climatology = np.mean(
                    [it["tokens"] for it in clim_items], axis=0
                ).astype(np.float32)
            else:
                climatology = past_items[-1]["tokens"].astype(np.float32)

            past = np.stack([p["tokens"] for p in past_items], axis=0)
            samples.append(
                ForecastSample(
                    past=past.astype(np.float32),
                    target=target_item["tokens"].astype(np.float32),
                    seasonal=seasonal,
                    delta_days=deltas,
                    doy_sincos=_doy_encoding(t_target),
                    location_id=loc_id,
                    past_keys=[p["key"] for p in past_items],
                    target_key=target_item["key"],
                    has_seasonal=has_seasonal,
                    seasonal_key=seasonal_item["key"] if has_seasonal else None,
                    climatology=climatology,
                    has_climatology=has_climatology,
                    climatology_keys=[it["key"] for it in clim_items] or None,
                )
            )
    return samples


def nuisance_floor(
    sequences: dict[str, list[dict]],
    max_gap_days: float = 7.0,
    min_gap_days: float = 1.0,
) -> list[dict[str, Any]]:
    """Residual energy between clear scenes only a few days apart.

    Two cloud-free observations five days apart contain almost no real
    land-surface change, so the residual between them is dominated by things no
    forecaster can predict: atmospheric state, illumination and view geometry,
    residual haze, sub-pixel co-registration. That number is the measurement
    noise of the whole pipeline and it is the scale every forecasting result
    has to be read against.

    It is an **upper bound** on the nuisance: a few days of genuine phenological
    change are baked in, especially over fast-growing crops.

    One row per pair, with ``residual_mse`` directly comparable to the
    ``residual_mse`` reported by :func:`evaluate_forecaster`.
    """
    rows: list[dict[str, Any]] = []
    for loc_id, seq in sequences.items():
        ordered = sorted(seq, key=_item_datetime)
        for i in range(len(ordered) - 1):
            for j in range(i + 1, len(ordered)):
                gap = (_item_datetime(ordered[j]) - _item_datetime(ordered[i])).days
                if gap > max_gap_days:
                    break
                if gap < min_gap_days:
                    continue
                a = ordered[i]["tokens"].astype(np.float32)
                b = ordered[j]["tokens"].astype(np.float32)
                rows.append(
                    {
                        "location_id": loc_id,
                        "key_a": ordered[i]["key"],
                        "key_b": ordered[j]["key"],
                        "gap_days": int(gap),
                        "residual_mse": float(np.mean((b - a) ** 2)),
                        "mean_change": float(np.mean(1.0 - cosine_similarity_map(a, b))),
                    }
                )
    return rows


def change_sample_weights(samples: list[ForecastSample], eps: float = 1e-6) -> torch.Tensor:
    """Upweight windows where the land actually moved (residual energy)."""
    w = []
    for s in samples:
        w.append(float(np.mean((s.target - s.past[-1]) ** 2)))
    arr = np.asarray(w, dtype=np.float64) + eps
    arr = arr / arr.sum()
    return torch.as_tensor(arr, dtype=torch.double)


def split_samples_by_year(
    samples: list[ForecastSample],
    train_years: tuple[int, ...] = (2022, 2023),
    test_years: tuple[int, ...] = (2024,),
) -> tuple[list[ForecastSample], list[ForecastSample], list[ForecastSample]]:
    """Split on the *target* date year. Leftover years go to validation."""

    def year_of(s: ForecastSample) -> int:
        return int(s.target_key[:4])

    train, val, test = [], [], []
    for s in samples:
        y = year_of(s)
        if y in test_years:
            test.append(s)
        elif y in train_years:
            train.append(s)
        else:
            val.append(s)
    if not train and samples:
        ordered = sorted(samples, key=lambda s: s.target_key)
        n = len(ordered)
        n_train = max(1, int(0.6 * n))
        n_val = max(1, int(0.2 * n)) if n >= 5 else 0
        train = ordered[:n_train]
        val = ordered[n_train : n_train + n_val] if n_val else ordered[n_train:]
        test = ordered[n_train + n_val :] if n_val else []
        if not test:
            test = val or train[-max(1, n // 5) :]
    if not val and train:
        n = max(1, len(train) // 5)
        val = train[-n:]
        train = train[:-n] or train
    return train, val, test


def split_samples_by_date(
    samples: list[ForecastSample],
    val_start: str,
    test_start: str,
) -> tuple[list[ForecastSample], list[ForecastSample], list[ForecastSample]]:
    """Chronological split on the *target* date, at two ``YYYY-MM-DD`` cutoffs.

    Needed wherever the signal is a trend rather than a repeating cycle. A
    split by calendar year would put an entire disturbance-recovery trajectory
    on one side of the boundary, so the model would be asked to extrapolate a
    process it had never seen. Splitting mid-trend lets it learn the recovery
    and then be tested on its continuation, which is the question actually
    being asked.
    """
    train, val, test = [], [], []
    for s in samples:
        key = s.target_key[:10]
        if key >= test_start:
            test.append(s)
        elif key >= val_start:
            val.append(s)
        else:
            train.append(s)
    return train, val, test


class ForecastTokenDataset(Dataset):
    def __init__(self, samples: list[ForecastSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        climatology = s.climatology
        if climatology is None:
            climatology = s.past[-1]
        return {
            "past": torch.from_numpy(s.past),
            "target": torch.from_numpy(s.target),
            "seasonal": torch.from_numpy(s.seasonal),
            "climatology": torch.from_numpy(np.asarray(climatology, dtype=np.float32)),
            "delta_days": torch.from_numpy(s.delta_days),
            "doy_sincos": torch.from_numpy(s.doy_sincos),
        }


# ---------------------------------------------------------------------------
# Models (residual: output Δz, add to last frame)
# ---------------------------------------------------------------------------


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
        )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTMForecaster(nn.Module):
    """Predict Δz; ẑ = z_t + Δz so the forecast stays near a real token."""

    def __init__(
        self,
        embed_dim: int = 768,
        proj_dim: int = 128,
        hidden: int = 96,
        extra_channels: int = 3,
        kernel_size: int = 3,
        residual: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.hidden = hidden
        self.residual = residual
        in_ch = proj_dim + extra_channels
        self.in_proj = nn.Linear(embed_dim, proj_dim)
        self.cell = ConvLSTMCell(in_ch, hidden, kernel_size=kernel_size)
        self.head = nn.Sequential(
            nn.Conv2d(hidden, proj_dim, kernel_size=1),
            nn.GELU(),
        )
        self.out_proj = nn.Linear(proj_dim, embed_dim)
        if residual:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def _extras(
        self,
        delta_days: torch.Tensor,
        doy_sincos: torch.Tensor,
        t_index: int,
        spatial: tuple[int, int],
    ) -> torch.Tensor:
        b = delta_days.shape[0]
        h, w = spatial
        dt = (delta_days[:, t_index] / 365.0).clamp(0.0, 2.0).view(b, 1, 1, 1)
        doy = doy_sincos.view(b, 2, 1, 1)
        extras = torch.cat([dt, doy], dim=1)
        return extras.expand(b, 3, h, w)

    def forward(
        self,
        past: torch.Tensor,
        delta_days: torch.Tensor,
        doy_sincos: torch.Tensor,
    ) -> torch.Tensor:
        b, k, h, w, d = past.shape
        x = self.in_proj(past)
        hid = past.new_zeros(b, self.hidden, h, w)
        cell = past.new_zeros(b, self.hidden, h, w)
        for t in range(k):
            frame = x[:, t].permute(0, 3, 1, 2)
            extra = self._extras(delta_days, doy_sincos, t, (h, w))
            inp = torch.cat([frame, extra], dim=1)
            hid, cell = self.cell(inp, hid, cell)
        feat = self.head(hid).permute(0, 2, 3, 1)
        delta = self.out_proj(feat)
        if self.residual:
            return past[:, -1] + delta
        return delta


class LinearDOYForecaster(nn.Module):
    """Roadmap baseline 4.1.3: z_t + Δt + day-of-year → ẑ.

    No temporal memory (only the last frame) and no spatial mixing. If this
    matches ConvLSTM, the sequence model is not what produces the skill.
    Set ``hidden=0`` for a pure linear head, >0 for a tiny MLP.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        proj_dim: int = 128,
        hidden: int = 0,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual
        self.in_proj = nn.Linear(embed_dim, proj_dim)
        in_features = proj_dim + 3
        if hidden and hidden > 0:
            self.head = nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.GELU(),
                nn.Linear(hidden, embed_dim),
            )
            last_layer = self.head[-1]
        else:
            self.head = nn.Linear(in_features, embed_dim)
            last_layer = self.head
        if residual:
            nn.init.zeros_(last_layer.weight)
            nn.init.zeros_(last_layer.bias)

    def forward(
        self,
        past: torch.Tensor,
        delta_days: torch.Tensor,
        doy_sincos: torch.Tensor,
    ) -> torch.Tensor:
        b, k, h, w, d = past.shape
        last = past[:, -1]
        x = self.in_proj(last)
        dt = (delta_days[:, -1] / 365.0).clamp(0.0, 2.0).view(b, 1, 1, 1).expand(b, h, w, 1)
        doy = doy_sincos.view(b, 1, 1, 2).expand(b, h, w, 2)
        delta = self.head(torch.cat([x, dt, doy], dim=-1))
        if self.residual:
            return last + delta
        return delta


class TokenLSTMForecaster(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        proj_dim: int = 128,
        hidden: int = 96,
        residual: bool = True,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.residual = residual
        self.in_proj = nn.Linear(embed_dim, proj_dim)
        self.d_in = proj_dim + 3
        self.lstm = nn.LSTM(self.d_in, hidden, batch_first=True)
        self.out = nn.Linear(hidden, embed_dim)
        if residual:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(
        self,
        past: torch.Tensor,
        delta_days: torch.Tensor,
        doy_sincos: torch.Tensor,
    ) -> torch.Tensor:
        b, k, h, w, d = past.shape
        n = h * w
        x = self.in_proj(past).reshape(b, k, n, self.proj_dim)
        dt = (delta_days / 365.0).clamp(0.0, 2.0).unsqueeze(-1).unsqueeze(2)
        dt = dt.expand(b, k, n, 1)
        doy = doy_sincos.view(b, 1, 1, 2).expand(b, k, n, 2)
        seq = torch.cat([x, dt, doy], dim=-1)
        seq = seq.permute(0, 2, 1, 3).reshape(b * n, k, self.d_in)
        out, _ = self.lstm(seq)
        delta = self.out(out[:, -1]).reshape(b, h, w, d)
        if self.residual:
            return past[:, -1] + delta
        return delta


# ---------------------------------------------------------------------------
# Losses, metrics, train
# ---------------------------------------------------------------------------


def token_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_n = F.normalize(pred, dim=-1)
    tgt_n = F.normalize(target, dim=-1)
    return (1.0 - (pred_n * tgt_n).sum(dim=-1)).mean()


def residual_mse(pred: torch.Tensor, target: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred - last, target - last)


def change_map_mse(pred: torch.Tensor, target: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
    """MSE between predicted and real 14×14 (1 - cosine) maps."""
    def _map(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_n = F.normalize(a, dim=-1)
        b_n = F.normalize(b, dim=-1)
        return 1.0 - (a_n * b_n).sum(dim=-1)

    return F.mse_loss(_map(pred, last), _map(target, last))


@torch.no_grad()
def batch_aligned_cosine(pred: torch.Tensor, target: torch.Tensor) -> float:
    p = pred.detach().cpu().numpy()
    t = target.detach().cpu().numpy()
    vals = [float(np.mean(cosine_similarity_map(p[i], t[i]))) for i in range(p.shape[0])]
    return float(np.mean(vals))


@torch.no_grad()
def batch_change_corr(
    pred: torch.Tensor,
    target: torch.Tensor,
    last: torch.Tensor,
    mask: np.ndarray | None = None,
) -> float:
    p = pred.detach().cpu().numpy()
    t = target.detach().cpu().numpy()
    l = last.detach().cpu().numpy()
    sel = None if mask is None else np.asarray(mask, dtype=bool).ravel()
    corrs = []
    for i in range(p.shape[0]):
        pc = (1.0 - cosine_similarity_map(p[i], l[i])).ravel()
        rc = (1.0 - cosine_similarity_map(t[i], l[i])).ravel()
        if sel is not None:
            pc, rc = pc[sel], rc[sel]
        if pc.size < 3 or pc.std() < 1e-8 or rc.std() < 1e-8:
            corrs.append(0.0)
            continue
        corrs.append(float(np.corrcoef(pc, rc)[0, 1]))
    return float(np.nanmean(corrs))


@torch.no_grad()
def trend_prediction(past: torch.Tensor, delta_days: torch.Tensor) -> torch.Tensor:
    """Per-token least-squares linear extrapolation of the history.

    The baseline that matters for post-disturbance recovery. Persistence assumes
    the surface stops changing and climatology assumes it repeats last year;
    neither can express "this pixel has been greening steadily for a year". A
    straight line through the history can, so a dynamics model that cannot beat
    it has not learned recovery, only that recovery is happening.

    ``delta_days[:, k]`` is days from ``past[:, k]`` to the target, so times
    relative to the target are ``-delta_days`` and the prediction is the fit
    evaluated at zero.
    """
    b, k = past.shape[0], past.shape[1]
    if k < 2:
        return past[:, -1]
    t = (-delta_days.to(past.dtype)).view(b, k, *([1] * (past.dim() - 2)))
    t_bar = t.mean(dim=1, keepdim=True)
    z_bar = past.mean(dim=1, keepdim=True)
    dt = t - t_bar
    denom = (dt * dt).sum(dim=1, keepdim=True)
    slope = (dt * (past - z_bar)).sum(dim=1, keepdim=True) / denom.clamp(min=1e-6)
    pred = (z_bar - slope * t_bar).squeeze(1)
    # Degenerate history (all frames same date) falls back to persistence.
    flat = denom.squeeze(1).squeeze(-1) <= 1e-6 if denom.dim() > 2 else denom.view(b) <= 1e-6
    if bool(flat.any()):
        pred = torch.where(flat.view(b, *([1] * (pred.dim() - 1))), past[:, -1], pred)
    return pred


def evaluate_forecaster(
    model: nn.Module | None,
    loader: DataLoader,
    device: torch.device,
    kind: str = "model",
    mask: np.ndarray | None = None,
    reference: float | None = None,
) -> dict[str, float]:
    """kind: 'model' | 'persistence' | 'seasonal' | 'climatology' | 'trend'.

    ``mask`` is an optional ``(H, W)`` boolean grid restricting every metric to a
    subset of tokens — used to score inside and outside a burn scar separately.
    ``reference`` is a residual MSE to normalise against (the per-window win rate
    and MSE ratio are only meaningful relative to something).
    """
    cos_pred, res_pred, cmap = [], [], []
    if model is not None:
        model.eval()
    sel = None if mask is None else torch.from_numpy(np.asarray(mask, dtype=bool))
    n = 0
    for batch in loader:
        past = batch["past"].to(device)
        target = batch["target"].to(device)
        last = past[:, -1]
        if kind == "persistence":
            pred = last
        elif kind == "seasonal":
            pred = batch["seasonal"].to(device)
        elif kind == "climatology":
            pred = batch["climatology"].to(device)
        elif kind == "trend":
            pred = trend_prediction(past, batch["delta_days"].to(device))
        else:
            assert model is not None
            with torch.no_grad():
                pred = model(
                    past,
                    batch["delta_days"].to(device),
                    batch["doy_sincos"].to(device),
                )
        if sel is not None:
            m = sel.to(pred.device)
            p_m, t_m, l_m = pred[:, m], target[:, m], last[:, m]
        else:
            p_m, t_m, l_m = pred, target, last
        cos_pred.append(batch_aligned_cosine(p_m, t_m))
        res_pred.append(float(residual_mse(p_m, t_m, l_m).cpu()))
        cmap.append(batch_change_corr(pred, target, last, mask=mask))
        n += 1
    out = {
        "aligned_cosine": float(np.mean(cos_pred)) if cos_pred else float("nan"),
        "residual_mse": float(np.mean(res_pred)) if res_pred else float("nan"),
        "change_map_corr": float(np.mean(cmap)) if cmap else float("nan"),
        "n_batches": n,
    }
    if reference is not None and reference > 0:
        out["mse_ratio"] = out["residual_mse"] / reference
    return out


def per_window_residuals(
    model: nn.Module | None,
    loader: DataLoader,
    device: torch.device,
    kind: str = "model",
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Residual MSE for every window separately, in loader order.

    Paired per-window comparison is more robust than comparing two means, which
    a handful of bad windows can dominate. Feed two of these to
    :func:`win_rate`.
    """
    sel = None if mask is None else torch.from_numpy(np.asarray(mask, dtype=bool))
    if model is not None:
        model.eval()
    out: list[float] = []
    for batch in loader:
        past = batch["past"].to(device)
        target = batch["target"].to(device)
        last = past[:, -1]
        if kind == "persistence":
            pred = last
        elif kind == "seasonal":
            pred = batch["seasonal"].to(device)
        elif kind == "climatology":
            pred = batch["climatology"].to(device)
        elif kind == "trend":
            pred = trend_prediction(past, batch["delta_days"].to(device))
        else:
            assert model is not None
            with torch.no_grad():
                pred = model(
                    past,
                    batch["delta_days"].to(device),
                    batch["doy_sincos"].to(device),
                )
        if sel is not None:
            m = sel.to(pred.device)
            pred, target, last = pred[:, m], target[:, m], last[:, m]
        for i in range(pred.shape[0]):
            out.append(float(residual_mse(pred[i : i + 1], target[i : i + 1], last[i : i + 1])))
    return np.asarray(out, dtype=np.float64)


def win_rate(model_res: np.ndarray, baseline_res: np.ndarray) -> float:
    """Fraction of windows where the model has the lower residual."""
    n = min(len(model_res), len(baseline_res))
    if n == 0:
        return float("nan")
    return float(np.mean(model_res[:n] < baseline_res[:n]))


def train_forecaster(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    epochs: int = 40,
    lr: float = 1e-3,
    residual_weight: float = 1.0,
    cosine_weight: float = 0.2,
    change_map_weight: float = 0.5,
    ckpt_path: Path | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    emit = log or print
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val = float("inf")
    history: list[dict] = []
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            past = batch["past"].to(device)
            target = batch["target"].to(device)
            pred = model(
                past,
                batch["delta_days"].to(device),
                batch["doy_sincos"].to(device),
            )
            last = past[:, -1]
            loss = (
                cosine_weight * token_cosine_loss(pred, target)
                + residual_weight * residual_mse(pred, target, last)
                + change_map_weight * change_map_mse(pred, target, last)
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else float("nan")}
        if val_loader is not None and len(val_loader.dataset) > 0:
            metrics = evaluate_forecaster(model, val_loader, device, kind="model")
            row.update({f"val_{k}": v for k, v in metrics.items()})
            score = metrics["residual_mse"]
            if score < best_val:
                best_val = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(row)
        extra = ""
        if "val_residual_mse" in row:
            extra = (
                f"  val_res_mse={row['val_residual_mse']:.4f}"
                f"  val_cmap={row.get('val_change_map_corr', float('nan')):.3f}"
            )
        emit(f"epoch {epoch:03d}  train_loss={row['train_loss']:.4f}{extra}")

    if best_state is not None:
        model.load_state_dict(best_state)
    if ckpt_path is not None:
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "history": history}, ckpt_path)
        emit(f"saved checkpoint → {ckpt_path}")
    return {"history": history, "model": model}


# ---------------------------------------------------------------------------
# Multi-step rollout — is the latent iterable?
# ---------------------------------------------------------------------------


def build_rollout_chains(
    sequences: dict[str, list[dict]],
    history: int = 6,
    min_delta_days: float = 21.0,
    steps: int = 6,
    stride: int = 3,
) -> list[dict[str, Any]]:
    """Chains of successive real observations for autoregressive evaluation.

    A one-step forecaster and a world model are not the same claim. The world
    model claim requires that predicted states can be fed back in and stepped
    again, so this builds ``steps`` successive targets, each at least
    ``min_delta_days`` after the previous one, from a single real history.

    ``stride`` subsamples the chain start indices so neighbouring chains do not
    overlap almost completely.
    """
    chains: list[dict[str, Any]] = []
    for loc_id, seq in sequences.items():
        ordered = sorted(seq, key=_item_datetime)
        n = len(ordered)
        for last_idx in range(history - 1, n - 1, stride):
            past_items = ordered[last_idx - history + 1 : last_idx + 1]
            targets, cursor = [], _item_datetime(past_items[-1])
            k = last_idx + 1
            while k < n and len(targets) < steps:
                if (_item_datetime(ordered[k]) - cursor).days >= min_delta_days:
                    targets.append(ordered[k])
                    cursor = _item_datetime(ordered[k])
                k += 1
            if len(targets) < 2:
                continue
            chains.append(
                {
                    "location_id": loc_id,
                    "past": np.stack([p["tokens"] for p in past_items], axis=0).astype(np.float32),
                    "past_dates": [_item_datetime(p) for p in past_items],
                    "past_keys": [p["key"] for p in past_items],
                    "targets": [t["tokens"].astype(np.float32) for t in targets],
                    "target_dates": [_item_datetime(t) for t in targets],
                    "target_keys": [t["key"] for t in targets],
                }
            )
    return chains


@torch.no_grad()
def _predict_at(
    model: nn.Module,
    hist: list[np.ndarray],
    dates: list[datetime],
    target_date: datetime,
    device: torch.device,
) -> np.ndarray:
    past = np.stack(hist, axis=0)[None]
    deltas = np.array([[(target_date - d).days for d in dates]], dtype=np.float32)
    doy = _doy_encoding(target_date)[None]
    pred = model(
        torch.from_numpy(past).to(device),
        torch.from_numpy(deltas).to(device),
        torch.from_numpy(doy).to(device),
    )
    return pred[0].detach().cpu().numpy()


@torch.no_grad()
def rollout_evaluate(
    model: nn.Module,
    chains: list[dict[str, Any]],
    device: torch.device,
) -> list[dict[str, Any]]:
    """Score autoregressive rollout against direct prediction and persistence.

    Everything is measured relative to the **last real observation**, so step *n*
    is directly comparable to a persistence forecast over the same total horizon.

    The comparison that matters is ``rollout`` vs ``direct``. Both predict the
    same date; only the rollout has to consume its own output. If rollout error
    grows much faster, the predicted states are drifting off the manifold the
    encoder produces and the "world model" framing is not earned.

    ``norm_ratio`` is the diagnostic for that drift: a rollout that collapses
    toward the mean shows a ratio falling below 1.
    """
    model.eval()
    rows: list[dict[str, Any]] = []
    for ci, ch in enumerate(chains):
        anchor = ch["past"][-1]
        base_hist = [f for f in ch["past"]]
        hist, dates = list(base_hist), list(ch["past_dates"])
        t0 = ch["past_dates"][-1]
        for step, (real, tdate, tkey) in enumerate(
            zip(ch["targets"], ch["target_dates"], ch["target_keys"]), start=1
        ):
            pred = _predict_at(model, hist[-len(base_hist):], dates[-len(base_hist):], tdate, device)
            direct = _predict_at(model, base_hist, ch["past_dates"], tdate, device)
            real_delta = real - anchor
            rows.append(
                {
                    "chain": ci,
                    "location_id": ch["location_id"],
                    "step": step,
                    "target_key": tkey,
                    "horizon_days": int((tdate - t0).days),
                    "rollout_mse": float(np.mean((pred - anchor - real_delta) ** 2)),
                    "direct_mse": float(np.mean((direct - anchor - real_delta) ** 2)),
                    "persistence_mse": float(np.mean(real_delta**2)),
                    "rollout_cosine": float(np.mean(cosine_similarity_map(pred, real))),
                    "direct_cosine": float(np.mean(cosine_similarity_map(direct, real))),
                    "norm_ratio": float(
                        np.linalg.norm(pred) / max(np.linalg.norm(real), 1e-8)
                    ),
                }
            )
            hist.append(pred)
            dates.append(tdate)
    return rows


def summarize_rollout(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-step means plus skill of rollout and direct against persistence."""
    df: dict[int, list[dict]] = {}
    for r in rows:
        df.setdefault(int(r["step"]), []).append(r)
    out = []
    for step in sorted(df):
        g = df[step]
        pers = float(np.mean([x["persistence_mse"] for x in g]))
        roll = float(np.mean([x["rollout_mse"] for x in g]))
        direct = float(np.mean([x["direct_mse"] for x in g]))
        out.append(
            {
                "step": step,
                "n_chains": len(g),
                "mean_horizon_days": float(np.mean([x["horizon_days"] for x in g])),
                "persistence_mse": pers,
                "rollout_mse": roll,
                "direct_mse": direct,
                "rollout_skill": 1.0 - roll / pers if pers else float("nan"),
                "direct_skill": 1.0 - direct / pers if pers else float("nan"),
                "rollout_cosine": float(np.mean([x["rollout_cosine"] for x in g])),
                "direct_cosine": float(np.mean([x["direct_cosine"] for x in g])),
                "norm_ratio": float(np.mean([x["norm_ratio"] for x in g])),
                "rollout_win_rate": float(
                    np.mean([x["rollout_mse"] < x["persistence_mse"] for x in g])
                ),
            }
        )
    return out


def change_maps_from_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    last: torch.Tensor,
) -> dict[str, np.ndarray]:
    pred_np = pred.detach().cpu().numpy()[0]
    tgt_np = target.detach().cpu().numpy()[0]
    last_np = last.detach().cpu().numpy()[0]
    return {
        "predicted_change": 1.0 - cosine_similarity_map(pred_np, last_np),
        "real_change": 1.0 - cosine_similarity_map(tgt_np, last_np),
        "error_vs_target": 1.0 - cosine_similarity_map(pred_np, tgt_np),
        "persistence_change": 1.0 - cosine_similarity_map(last_np, last_np),
    }


def make_weighted_loader(
    samples: list[ForecastSample],
    batch_size: int,
    seed: int | None = None,
) -> DataLoader:
    ds = ForecastTokenDataset(samples)
    if not samples:
        return DataLoader(ds, batch_size=1)
    weights = change_sample_weights(samples)
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(int(seed))
    sampler = WeightedRandomSampler(
        weights, num_samples=len(samples), replacement=True, generator=generator
    )
    return DataLoader(ds, batch_size=min(batch_size, len(samples)), sampler=sampler)


# ---------------------------------------------------------------------------
# Multi-seed evaluation
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_seeds(
    model_factories: dict[str, Callable[[], nn.Module]],
    train_samples: list[ForecastSample],
    val_samples: list[ForecastSample],
    test_samples: list[ForecastSample],
    device: torch.device,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    batch_size: int = 8,
    epochs: int = 40,
    lr: float = 1e-3,
    ckpt_dir: Path | None = None,
    log: Callable[[str], None] | None = None,
    **train_kwargs: Any,
) -> list[dict[str, Any]]:
    """Train every model once per seed and score it on the test windows.

    Differences between these architectures are small enough that a single run
    cannot separate them: re-training the *same* model with a different
    initialisation moves residual MSE by more than the gaps between models. One
    row per (model, seed); pair with :func:`summarize_seed_runs`.
    """
    emit = log or print
    val_loader = DataLoader(ForecastTokenDataset(val_samples), batch_size=1) if val_samples else None
    test_loader = DataLoader(ForecastTokenDataset(test_samples), batch_size=1)
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        for name, factory in model_factories.items():
            seed_everything(int(seed))
            model = factory()
            train_loader = make_weighted_loader(train_samples, batch_size, seed=int(seed))
            ckpt_path = None
            if ckpt_dir is not None and int(seed) == 0:
                ckpt_path = Path(ckpt_dir) / f"{name}_seed0.pt"
            train_forecaster(
                model,
                train_loader,
                val_loader,
                device,
                epochs=epochs,
                lr=lr,
                ckpt_path=ckpt_path,
                log=lambda _msg: None,
                **train_kwargs,
            )
            metrics = evaluate_forecaster(model, test_loader, device, kind="model")
            row = {"model": name, "seed": int(seed), **metrics}
            rows.append(row)
            emit(
                f"seed {seed}  {name:<24s} res_mse={metrics['residual_mse']:.4f} "
                f"cmap={metrics['change_map_corr']:.3f}"
            )
    return rows


def summarize_seed_runs(
    rows: list[dict[str, Any]],
    metrics: Sequence[str] = ("residual_mse", "change_map_corr", "aligned_cosine"),
) -> list[dict[str, Any]]:
    """Mean ± std per model, plus the spread that defines the noise floor."""
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)
    out = []
    for name, group in by_model.items():
        summary: dict[str, Any] = {"model": name, "n_seeds": len(group)}
        for metric in metrics:
            vals = np.array([g[metric] for g in group], dtype=float)
            summary[f"{metric}_mean"] = float(vals.mean())
            summary[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            summary[f"{metric}_min"] = float(vals.min())
            summary[f"{metric}_max"] = float(vals.max())
        out.append(summary)
    return out


BASELINE_KINDS: tuple[tuple[str, str], ...] = (
    ("persistence", "persistence"),
    ("seasonal copy", "seasonal"),
    ("climatology", "climatology"),
)

#: Adds linear extrapolation. Use where the signal has a trend (disturbance
#: recovery), not for pure phenology where a straight line is meaningless.
BASELINE_KINDS_TREND: tuple[tuple[str, str], ...] = BASELINE_KINDS + (("trend", "trend"),)


def baseline_rows(
    samples: list[ForecastSample],
    device: torch.device,
    kinds: Sequence[tuple[str, str]] = BASELINE_KINDS,
    mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Deterministic baselines, scored once on the windows the models are scored on.

    These have no parameters, so they get no seed spread.
    """
    loader = DataLoader(ForecastTokenDataset(samples), batch_size=1)
    rows = []
    for label, kind in kinds:
        rows.append(
            {"model": label, **evaluate_forecaster(None, loader, device, kind=kind, mask=mask)}
        )
    return rows
