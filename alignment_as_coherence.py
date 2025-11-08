#!/usr/bin/env python3
"""
alignment_as_coherence.py
Author: Robert C. Ventura

Alignment as Coherence — A Minimal Demonstration

Concept:
Model an AI system's "belief field" as a 1D reaction–diffusion process
with Fisher–KPP-style dynamics and a simple global self-correction term.

Not a production training loop. Serves as a physics-flavored alignment toy.

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_synthetic_world(
    n_points: int = 400,
    true_region_fraction: float = 0.6,
    flip_noise: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Create a 1D line of ground-truth labels in {-1, +1}."""
    rng = np.random.default_rng(seed)
    n_true = int(n_points * true_region_fraction)
    n_false = n_points - n_true
    left = np.ones(n_true)
    right = -np.ones(n_false)
    truth = np.concatenate([left, right])
    flip_mask = rng.random(n_points) < flip_noise
    truth[flip_mask] *= -1
    return truth


def initialize_beliefs(
    truth: np.ndarray,
    mismatch_region_fraction: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Initialize belief field u(x) in [0,1] with partially misaligned region."""
    rng = np.random.default_rng(seed)
    n = len(truth)
    cutoff = int(n * mismatch_region_fraction)
    beliefs = np.empty(n)
    beliefs[:cutoff] = 0.8 + 0.1 * rng.standard_normal(cutoff)
    beliefs[cutoff:] = 0.2 + 0.2 * rng.standard_normal(n - cutoff)
    return np.clip(beliefs, 0.01, 0.99)


def laplacian_1d(field: np.ndarray, dx: float) -> np.ndarray:
    """Neumann boundary 1D Laplacian."""
    lap = np.empty_like(field)
    lap[1:-1] = (field[2:] - 2 * field[1:-1] + field[:-2]) / (dx ** 2)
    lap[0] = (field[1] - field[0]) / (dx ** 2)
    lap[-1] = (field[-2] - field[-1]) / (dx ** 2)
    return lap


def update_beliefs(
    beliefs: np.ndarray,
    truth: np.ndarray,
    D: float,
    lam: float,
    dt: float,
    dx: float,
    honesty_boost: float,
    noise_scale: float,
    global_feedback_strength: float,
    rng: np.random.Generator,
):
    """One Euler step of coherence dynamics."""
    u = beliefs
    diff_term = D * laplacian_1d(u, dx)

    prob_correct = np.where(truth == 1, u, 1 - u)
    local_alignment = (prob_correct - 0.5) * 2.0
    lambda_local = lam * (1.0 + honesty_boost * local_alignment)

    global_coherence = float(np.mean(prob_correct))
    lambda_global = 1.0 + global_feedback_strength * (global_coherence - 0.5)
    lambda_eff = lambda_local * lambda_global

    reaction_term = lambda_eff * u * (1.0 - u)

    noise = noise_scale * rng.standard_normal(len(u))

    du = diff_term + reaction_term + noise
    u_next = np.clip(u + dt * du, 0.0, 1.0)
    return u_next, global_coherence, float(lambda_eff.mean())


def compute_alignment_metrics(beliefs: np.ndarray, truth: np.ndarray):
    pred_labels = np.where(beliefs >= 0.5, 1, -1)
    correct = (pred_labels == truth).astype(float)
    accuracy = float(correct.mean())
    mean_conf = float(np.mean(np.abs(beliefs - 0.5)))
    return accuracy, mean_conf


def run_simulation(
    n_points: int = 400,
    steps: int = 800,
    dx: float = 1.0,
    dt: float = 0.05,
    D: float = 0.7,
    lam: float = 0.3,
    honesty_boost: float = 1.2,
    noise_scale: float = 0.02,
    global_feedback_strength: float = 0.8,
    plot_interval: int = 80,
    seed: int = 7,
):
    rng = np.random.default_rng(seed)
    truth = generate_synthetic_world(n_points=n_points)
    beliefs = initialize_beliefs(truth)
    c_theoretical = 2.0 * np.sqrt(D * lam)

    acc_hist, coh_hist, lam_hist = [], [], []
    xs = np.arange(n_points)

    plt.ion()
    fig, (ax_field, ax_metrics) = plt.subplots(2, 1, figsize=(8, 6))
    fig.suptitle("Alignment as Coherence — Reaction–Diffusion Demo")

    for t in range(steps):
        beliefs, global_coh, lam_eff_mean = update_beliefs(
            beliefs, truth, D, lam, dt, dx,
            honesty_boost, noise_scale, global_feedback_strength, rng
        )
        acc, _ = compute_alignment_metrics(beliefs, truth)
        acc_hist.append(acc)
        coh_hist.append(global_coh)
        lam_hist.append(lam_eff_mean)

        if (t % plot_interval == 0) or (t == steps - 1):
            ax_field.clear()
            ax_metrics.clear()

            ax_field.plot(xs, (truth + 1) / 2, "--", label="Truth (scaled)")
            ax_field.plot(xs, beliefs, label="Belief field u(x)")
            ax_field.set_ylim(-0.1, 1.1)
            ax_field.set_ylabel("u(x)")
            ax_field.legend(loc="upper right")
            ax_field.set_title(f"Belief Field at step {t} — c ≈ {c_theoretical:.2f}")

            ax_metrics.plot(acc_hist, label="Accuracy")
            ax_metrics.plot(coh_hist, label="Global Coherence")
            ax_metrics.plot(lam_hist, label="Mean λ_eff")
            ax_metrics.set_ylim(0.0, 1.1)
            ax_metrics.set_xlabel("Step")
            ax_metrics.set_ylabel("Metric")
            ax_metrics.legend(loc="lower right")

            plt.pause(0.001)

    plt.ioff()
    plt.show()

    print("\n=== Alignment as Coherence — Summary ===")
    print(f"Final accuracy:          {acc_hist[-1]:.3f}")
    print(f"Final global coherence:  {coh_hist[-1]:.3f}")
    print(f"Front speed c=2√(Dλ):    {c_theoretical:.3f}")


if __name__ == "__main__":
    run_simulation()
