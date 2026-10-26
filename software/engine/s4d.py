"""Per-token S4D dynamics over frozen Prithvi tokens."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def interframe_and_horizon(delta_days: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if delta_days.ndim != 2:
        raise ValueError(f"delta_days must be (B, K), got {tuple(delta_days.shape)}")
    b, k = delta_days.shape
    inter = torch.ones(b, k, device=delta_days.device, dtype=delta_days.dtype)
    if k > 1:
        inter[:, 1:] = (delta_days[:, :-1] - delta_days[:, 1:]).clamp(min=1.0)
    horizon = delta_days[:, -1].clamp(min=1.0)
    return inter, horizon


def _zoh(A: torch.Tensor, delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dA = torch.exp(delta.unsqueeze(-1) * A)
    small = A.abs() < 1e-6
    safe = torch.where(small, torch.ones_like(A), A)
    coeff = (dA - 1) / safe
    coeff = torch.where(
        small.expand_as(coeff),
        delta.unsqueeze(-1) + 0.5 * delta.unsqueeze(-1) ** 2 * A,
        coeff,
    )
    return dA, coeff


class S4DRecurrent(nn.Module):
    def __init__(self, d_model: int, d_state: int = 32):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        n = torch.arange(d_state, dtype=torch.float32)
        tau = torch.logspace(math.log10(7.0), math.log10(365.0), d_state)
        self.log_A_real = nn.Parameter(torch.log(1.0 / tau))
        self.A_imag = nn.Parameter(2.0 * math.pi * n / 365.25)
        self.B = nn.Parameter(0.5 * torch.ones(d_model, d_state))
        self.C_real = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.C_imag = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.D = nn.Parameter(torch.zeros(d_model))

    def A(self) -> torch.Tensor:
        return -self.log_A_real.exp() + 1j * self.A_imag

    def timescales_days(self) -> dict[str, np.ndarray]:
        with torch.no_grad():
            A = self.A().detach().cpu()
            real = A.real.clamp(max=-1e-8)
            imag = A.imag.abs().clamp(min=1e-8)
            tau = -1.0 / real
            period = 2.0 * math.pi / imag
            c_mag = torch.sqrt(self.C_real**2 + self.C_imag**2).mean(dim=0).detach().cpu()
            total = float(c_mag.sum()) or 1.0
        return {
            "mode": np.arange(self.d_state),
            "tau_decay_days": tau.numpy(),
            "period_days": period.numpy(),
            "A_real": A.real.numpy(),
            "A_imag": A.imag.numpy(),
            "c_magnitude": c_mag.numpy(),
            "c_share": (c_mag / total).numpy(),
        }

    def step(self, x, h, delta):
        A = self.A().to(dtype=torch.complex64)
        dA, coeff = _zoh(A, delta)
        dA = dA.unsqueeze(1)
        coeff = coeff.unsqueeze(1)
        Bx = x.unsqueeze(-1).to(h.dtype) * self.B.to(h.dtype)
        h = dA * h + coeff * Bx
        C = torch.complex(self.C_real, self.C_imag).to(h.dtype)
        y = 2.0 * (h * C).sum(dim=-1).real + self.D * x
        return h, y


class TokenS4DForecaster(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        proj_dim: int = 128,
        d_state: int = 32,
        residual: bool = True,
        constant_delta: float | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.d_state = d_state
        self.residual = residual
        self.constant_delta = constant_delta
        self.d_model = proj_dim + 2
        self.in_proj = nn.Linear(embed_dim, proj_dim)
        self.ssm = S4DRecurrent(self.d_model, d_state=d_state)
        self.out = nn.Linear(self.d_model, embed_dim)
        if residual:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def _delta_schedule(self, delta_days, mode, constant_delta):
        inter, horizon = interframe_and_horizon(delta_days)
        if mode == "true":
            return inter, horizon
        if mode == "unit":
            return torch.ones_like(inter), torch.ones_like(horizon)
        if mode == "constant":
            c = constant_delta if constant_delta is not None else self.constant_delta
            if c is None:
                raise ValueError("delta_mode='constant' needs an explicit constant_delta")
            c = max(float(c), 1.0)
            return torch.full_like(inter, c), torch.full_like(horizon, c)
        raise ValueError(f"delta_mode must be 'true', 'unit', or 'constant', got {mode!r}")

    def forward(
        self,
        past,
        delta_days,
        doy_sincos,
        delta_mode: str = "true",
        constant_delta: float | None = None,
    ):
        b, k, h, w, d = past.shape
        n = h * w
        tokens = self.in_proj(past).reshape(b, k, n, self.proj_dim)
        doy = doy_sincos.view(b, 1, 1, 2).expand(b, k, n, 2)
        seq = torch.cat([tokens, doy], dim=-1)
        seq = seq.permute(0, 2, 1, 3).reshape(b * n, k, self.d_model)
        inter, horizon = self._delta_schedule(delta_days, delta_mode, constant_delta)
        inter = inter.unsqueeze(1).expand(b, n, k).reshape(b * n, k)
        horizon = horizon.unsqueeze(1).expand(b, n).reshape(b * n)
        state = seq.new_zeros(b * n, self.d_model, self.d_state, dtype=torch.complex64)
        y = seq[:, -1]
        for t in range(k):
            state, y = self.ssm.step(seq[:, t], state, inter[:, t])
        state, y = self.ssm.step(seq[:, -1], state, horizon)
        delta = self.out(y).reshape(b, h, w, d)
        if self.residual:
            return past[:, -1] + delta
        return delta
