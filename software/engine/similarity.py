"""Token similarity: aligned cosine, shuffle control, pooling, CLS."""

from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (denom + eps)


def cosine_similarity_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim == 1 and b.ndim == 3:
        a = a.reshape(1, 1, -1)
    if b.ndim == 1 and a.ndim == 3:
        b = b.reshape(1, 1, -1)
    a_n = l2_normalize(a, axis=-1)
    b_n = l2_normalize(b, axis=-1)
    return np.sum(a_n * b_n, axis=-1)


def similarity_stats(sim: np.ndarray) -> dict[str, float]:
    sim = np.asarray(sim, dtype=np.float64).ravel()
    return {
        "mean": float(np.mean(sim)),
        "std": float(np.std(sim)),
        "p10": float(np.percentile(sim, 10)),
        "p50": float(np.percentile(sim, 50)),
        "p90": float(np.percentile(sim, 90)),
        "min": float(np.min(sim)),
        "max": float(np.max(sim)),
    }


def mean_token_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim == 4:
        a = a.mean(axis=(1, 2))
    elif a.ndim == 3:
        a = a.mean(axis=(0, 1))
    if b.ndim == 4:
        b = b.mean(axis=(1, 2))
    elif b.ndim == 3:
        b = b.mean(axis=(0, 1))
    return float(cosine_similarity_map(a, b))


def aligned_token_mean_cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(cosine_similarity_map(a, b)))


def aligned_similarity(a: np.ndarray, b: np.ndarray) -> dict:
    sim_map = cosine_similarity_map(a, b)
    stats = similarity_stats(sim_map)
    flat = sim_map.ravel()
    stats["p25"] = float(np.percentile(flat, 25))
    stats["p75"] = float(np.percentile(flat, 75))
    stats["median"] = stats["p50"]
    return {"stats": stats, "similarity_map": sim_map, "change_map": 1.0 - sim_map}


def mean_pool_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return mean_token_cosine(a, b)


def shuffle_similarity(
    a: np.ndarray,
    b: np.ndarray,
    n_shuffles: int = 100,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    h, w, d = b.shape
    vals = []
    b_flat = b.reshape(h * w, d)
    for _ in range(n_shuffles):
        perm = rng.permutation(h * w)
        b_shuf = b_flat[perm].reshape(h, w, d)
        vals.append(aligned_token_mean_cosine(a, b_shuf))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_shuffles": n_shuffles,
        "seed": seed,
    }


def cls_similarity(cls_a: np.ndarray, cls_b: np.ndarray) -> dict:
    a = np.asarray(cls_a, dtype=np.float64).ravel()
    b = np.asarray(cls_b, dtype=np.float64).ravel()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    l2 = float(np.linalg.norm(a - b))
    return {
        "cls_cosine": cos,
        "cls_l2": l2,
        "cls_a_norm": float(np.linalg.norm(a)),
        "cls_b_norm": float(np.linalg.norm(b)),
    }


def compare_token_pair(
    a: np.ndarray,
    b: np.ndarray,
    cls_a: np.ndarray | None = None,
    cls_b: np.ndarray | None = None,
    n_shuffles: int = 100,
    seed: int = 0,
    correspondence: str = "temporal_aligned",
) -> dict:
    aligned = aligned_similarity(a, b)
    out = {
        "correspondence": correspondence,
        "aligned_mean": aligned["stats"]["mean"],
        "aligned_std": aligned["stats"]["std"],
        "aligned_min": aligned["stats"]["min"],
        "aligned_max": aligned["stats"]["max"],
        "aligned_p10": aligned["stats"]["p10"],
        "aligned_p25": aligned["stats"]["p25"],
        "aligned_median": aligned["stats"]["median"],
        "aligned_p75": aligned["stats"]["p75"],
        "aligned_p90": aligned["stats"]["p90"],
        "mean_pool": mean_pool_similarity(a, b),
        "similarity_map": aligned["similarity_map"],
        "change_map": aligned["change_map"],
    }
    shuf = shuffle_similarity(a, b, n_shuffles=n_shuffles, seed=seed)
    out["shuffle_mean"] = shuf["mean"]
    out["shuffle_std"] = shuf["std"]
    out["shuffle_p10"] = shuf["p10"]
    out["shuffle_p90"] = shuf["p90"]
    if cls_a is not None and cls_b is not None:
        out.update(cls_similarity(cls_a, cls_b))
    return out
