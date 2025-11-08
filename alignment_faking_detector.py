#!/usr/bin/env python3
"""
alignment_faking_detector.py

Coherence-Based Alignment Faking Detector (Toy Model)

Models:
- v_deep(x): original internal values
- v_target(x): external training objective
- v_surface(x): post-training outward behavior

Detects:
- "Hidden preference islands" where surface != deep
- Measures "coherence depth": which attractor is dynamically deeper

Conceptual lens on deceptive alignment using continuous fields.
"""

import numpy as np
import matplotlib.pyplot as plt


def laplacian_1d(field: np.ndarray, dx: float) -> np.ndarray:
    """Simple Neumann 1D Laplacian."""
    lap = np.empty_like(field)
    lap[1:-1] = (field[2:] - 2 * field[1:-1] + field[:-2]) / (dx ** 2)
    lap[0] = (field[1] - field[0]) / (dx ** 2)
    lap[-1] = (field[-2] - field[-1]) / (dx ** 2)
    return lap


def init_value_fields(
    n_points: int = 400,
    seed: int = 42,
):
    """Initialize v_deep, v_target, v_surface."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, 1, n_points)

    v_deep = np.tanh(3 * (0.5 - np.abs(xs - 0.3)))
    v_deep = np.clip(v_deep, -1.0, 1.0)

    v_target = np.where(xs < 0.5, 1.0, -1.0)

    v_surface = 0.5 * v_deep + 0.5 * v_target
    v_surface += 0.2 * rng.standard_normal(n_points)
    v_surface = np.clip(v_surface, -1.0, 1.0)

    return xs, v_deep, v_target, v_surface


def update_surface_values(
    v_surface: np.ndarray,
    v_deep: np.ndarray,
    v_target: np.ndarray,
    dx: float,
    dt: float,
    D_surface: float = 0.4,
    coupling_to_deep: float = 0.3,
    training_strength: float = 0.6,
    noise_scale: float = 0.02,
    rng: np.random.Generator = None,
):
    """One step of diffusion + coupling + training + noise."""
    if rng is None:
        rng = np.random.default_rng()

    diff_term = D_surface * laplacian_1d(v_surface, dx)
    deep_term = coupling_to_deep * (v_deep - v_surface)
    train_term = training_strength * (v_target - v_surface)
    noise_term = noise_scale * rng.standard_normal(len(v_surface))

    dv = diff_term + deep_term + train_term + noise_term
    v_next = np.clip(v_surface + dt * dv, -1.0, 1.0)
    return v_next


def detect_hidden_islands(
    v_deep: np.ndarray,
    v_surface: np.ndarray,
    agreement_threshold: float = 0.4,
    min_island_size: int = 5,
):
    """Detect intervals where |v_deep - v_surface| > threshold."""
    mismatch = np.abs(v_deep - v_surface)
    mask = mismatch > agreement_threshold

    islands = []
    in_island = False
    start = 0

    for i, active in enumerate(mask):
        if active and not in_island:
            in_island = True
            start = i
        elif not active and in_island:
            end = i
            if end - start >= min_island_size:
                islands.append((start, end))
            in_island = False

    if in_island:
        end = len(mask)
        if end - start >= min_island_size:
            islands.append((start, end))

    return islands, mismatch


def measure_coherence_depth(
    base_field: np.ndarray,
    alt_field: np.ndarray,
    steps: int = 60,
    dx: float = 1.0,
    dt: float = 0.05,
    D_probe: float = 0.4,
    noise_scale: float = 0.08,
    rng: np.random.Generator = None,
):
    """Test which attractor is deeper under neutral dynamics."""
    if rng is None:
        rng = np.random.default_rng()

    probe = np.clip(
        alt_field + noise_scale * rng.standard_normal(len(alt_field)),
        -1.0,
        1.0,
    )

    def neutral_step(f):
        diff = D_probe * laplacian_1d(f, dx)
        noise = noise_scale * 0.3 * rng.standard_normal(len(f))
        return np.clip(f + dt * (diff + noise), -1.0, 1.0)

    for _ in range(steps):
        probe = neutral_step(probe)

    d_base = np.mean(np.abs(probe - base_field))
    d_alt = np.mean(np.abs(probe - alt_field))

    depth_score = 1.0 - d_alt / (d_base + d_alt + 1e-8)
    return float(depth_score)


def run_alignment_faking_demo(
    n_points: int = 400,
    train_steps: int = 300,
    dx: float = 1.0,
    dt: float = 0.05,
    seed: int = 7,
):
    rng = np.random.default_rng(seed)
    xs, v_deep, v_target, v_surface = init_value_fields(n_points, seed)

    for _ in range(train_steps):
        v_surface = update_surface_values(
            v_surface, v_deep, v_target,
            dx, dt,
            D_surface=0.4,
            coupling_to_deep=0.25,
            training_strength=0.7,
            noise_scale=0.03,
            rng=rng,
        )

    islands, mismatch = detect_hidden_islands(v_deep, v_surface)
    depth_score = measure_coherence_depth(
        base_field=v_deep,
        alt_field=v_surface,
        steps=80,
        dx=dx,
        dt=0.05,
        D_probe=0.5,
        noise_scale=0.08,
        rng=rng,
    )

    plt.figure(figsize=(10, 7))

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(xs, v_deep, label="Deep values", linewidth=2)
    ax1.plot(xs, v_target, "--", label="Training target", linewidth=1)
    ax1.plot(xs, v_surface, label="Surface values", linewidth=2)
    for (s, e) in islands:
        ax1.axvspan(xs[s], xs[e - 1], color="grey", alpha=0.15)
    ax1.set_ylabel("Value")
    ax1.set_title("Deep vs Surface Values & Hidden Islands")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(xs, mismatch, label="|Deep - Surface|", linewidth=2)
    ax2.axhline(0.4, linestyle="--", linewidth=1)
    for (s, e) in islands:
        ax2.axvspan(xs[s], xs[e - 1], color="grey", alpha=0.15)
    ax2.set_xlabel("Position")
    ax2.set_ylabel("Mismatch")
    ax2.set_title("Candidate Alignment-Faking Regions")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("COHERENCE-BASED ALIGNMENT FAKING DETECTOR — SUMMARY")
    print("=" * 60)
    print(f"Number of detected islands: {len(islands)}")
    for i, (s, e) in enumerate(islands):
        print(f"  Island {i+1}: indices [{s}, {e}), size={e-s}")
    print(f"Coherence depth score (>0.5 ⇒ deep prefs dominate): {depth_score:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    run_alignment_faking_demo()
