#!/usr/bin/env python3
"""
alignment_as_coherence_extended.py

Extended "Alignment as Coherence" demo with:

- Heterogeneous diffusion D(x)
- Multiple domains with varying ground-truth structure
- Reduced connectivity at domain boundaries
- Time-varying adversarial pressure
- Adaptive attacks targeting high-coherence regions

Concept:
Stress-test whether coherence-based dynamics can maintain alignment
across heterogeneous regions under targeted evolving attack.
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_heterogeneous_world(
    n_points: int = 400,
    n_domains: int = 4,
    flip_noise: float = 0.1,
    seed: int = 42,
):
    """World divided into domains with different truth distributions."""
    rng = np.random.default_rng(seed)
    truth = np.zeros(n_points, dtype=int)
    domains = np.zeros(n_points, dtype=int)

    domain_size = n_points // n_domains

    for d in range(n_domains):
        start = d * domain_size
        end = start + domain_size if d < n_domains - 1 else n_points
        domains[start:end] = d

        base_truth_prob = 0.8 - 0.15 * d
        base_truth_prob = np.clip(base_truth_prob, 0.05, 0.95)

        truth[start:end] = np.where(
            rng.random(end - start) < base_truth_prob,
            1,
            -1,
        )

    flip_mask = rng.random(n_points) < flip_noise
    truth[flip_mask] *= -1

    return truth, domains


def create_heterogeneous_diffusion(
    n_points: int,
    domains: np.ndarray,
    base_D: float = 0.7,
    domain_boundary_penalty: float = 0.3,
) -> np.ndarray:
    """Spatially-varying diffusion D(x) with reduced values near boundaries."""
    D = np.ones(n_points) * base_D
    boundaries = np.where(np.diff(domains) != 0)[0]
    for b in boundaries:
        for offset in range(-2, 3):
            idx = b + offset
            if 0 <= idx < n_points:
                D[idx] *= domain_boundary_penalty
    return D


class AdversarialPressure:
    """Time-varying adversarial field targeting belief coherence."""

    def __init__(
        self,
        n_points: int,
        base_strength: float = 0.1,
        attack_frequency: float = 0.05,
        seed: int = 0,
    ):
        self.n_points = n_points
        self.base_strength = base_strength
        self.attack_frequency = attack_frequency
        self.rng = np.random.default_rng(seed)

    def get_pressure(self, beliefs: np.ndarray, t: int) -> np.ndarray:
        time_factor = 1.0 + 0.3 * np.sin(t * self.attack_frequency)

        confidence = np.abs(beliefs - 0.5)
        if confidence.max() > 0:
            adaptive_targeting = confidence / confidence.max()
        else:
            adaptive_targeting = np.ones_like(beliefs)

        xs = np.arange(self.n_points)
        spatial_pattern = np.zeros(self.n_points)
        centers = self.rng.integers(0, self.n_points, size=3)
        for c in centers:
            dist = np.abs(xs - c)
            spatial_pattern += np.exp(-dist / 30.0)

        if spatial_pattern.max() > 0:
            spatial_pattern /= spatial_pattern.max()
        else:
            spatial_pattern += 1.0

        pressure_mag = (
            self.base_strength *
            time_factor *
            adaptive_targeting *
            spatial_pattern
        )

        noise = self.rng.standard_normal(self.n_points)

        return -pressure_mag * noise


def laplacian_1d_heterogeneous(
    field: np.ndarray,
    D: np.ndarray,
    dx: float,
) -> np.ndarray:
    """Heterogeneous Laplacian: div(D(x) * grad(field))."""
    n = len(field)
    lap = np.zeros_like(field)

    for i in range(1, n - 1):
        D_plus = 0.5 * (D[i] + D[i + 1])
        D_minus = 0.5 * (D[i] + D[i - 1])

        grad_plus = (field[i + 1] - field[i]) / dx
        grad_minus = (field[i] - field[i - 1]) / dx

        lap[i] = (D_plus * grad_plus - D_minus * grad_minus) / dx

    lap[0] = D[0] * (field[1] - field[0]) / (dx ** 2)
    lap[-1] = D[-1] * (field[-2] - field[-1]) / (dx ** 2)

    return lap


def update_beliefs_extended(
    beliefs: np.ndarray,
    truth: np.ndarray,
    D: np.ndarray,
    lam: float,
    dt: float,
    dx: float,
    honesty_boost: float,
    adversarial_pressure: np.ndarray,
    global_feedback_strength: float,
    rng: np.random.Generator,
):
    u = beliefs

    diff_term = laplacian_1d_heterogeneous(u, D, dx)

    prob_correct = np.where(truth == 1, u, 1 - u)
    local_alignment = (prob_correct - 0.5) * 2.0
    lambda_local = lam * (1.0 + honesty_boost * local_alignment)

    global_coherence = float(np.mean(prob_correct))
    feedback = np.tanh(global_feedback_strength * (global_coherence - 0.5))
    lambda_global = 1.0 + feedback

    lambda_eff = lambda_local * lambda_global

    reaction_term = lambda_eff * u * (1.0 - u)

    adv_term = adversarial_pressure

    du = diff_term + reaction_term + adv_term
    u_next = np.clip(u + dt * du, 0.0, 1.0)

    return u_next, global_coherence, float(lambda_eff.mean())


def run_extended_simulation(
    n_points: int = 400,
    n_domains: int = 4,
    steps: int = 1000,
    dx: float = 1.0,
    dt: float = 0.04,
    base_D: float = 0.8,
    lam: float = 0.4,
    honesty_boost: float = 1.5,
    adversarial_strength: float = 0.08,
    attack_frequency: float = 0.03,
    global_feedback_strength: float = 1.0,
    domain_boundary_penalty: float = 0.3,
    plot_interval: int = 50,
    seed: int = 7,
):
    rng = np.random.default_rng(seed)

    truth, domains = generate_heterogeneous_world(
        n_points=n_points,
        n_domains=n_domains,
        seed=seed,
    )

    D = create_heterogeneous_diffusion(
        n_points,
        domains,
        base_D=base_D,
        domain_boundary_penalty=domain_boundary_penalty,
    )

    adversary = AdversarialPressure(
        n_points=n_points,
        base_strength=adversarial_strength,
        attack_frequency=attack_frequency,
        seed=seed + 1,
    )

    beliefs = 0.5 + 0.2 * rng.standard_normal(n_points)
    beliefs = np.clip(beliefs, 0.01, 0.99)

    acc_hist, coh_hist, lam_hist, adv_hist = [], [], [], []
    xs = np.arange(n_points)

    plt.ion()
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    ax_field = fig.add_subplot(gs[0, :])
    ax_diff = fig.add_subplot(gs[1, 0])
    ax_adv = fig.add_subplot(gs[1, 1])
    ax_metrics = fig.add_subplot(gs[2, :])
    fig.suptitle("Alignment as Coherence — Extended Dynamics")

    for t in range(steps):
        adv_pressure = adversary.get_pressure(beliefs, t)

        beliefs, global_coh, lam_eff_mean = update_beliefs_extended(
            beliefs,
            truth,
            D,
            lam,
            dt,
            dx,
            honesty_boost,
            adv_pressure,
            global_feedback_strength,
            rng,
        )

        pred_labels = np.where(beliefs >= 0.5, 1, -1)
        accuracy = float((pred_labels == truth).mean())

        acc_hist.append(accuracy)
        coh_hist.append(global_coh)
        lam_hist.append(lam_eff_mean)
        adv_hist.append(float(np.abs(adv_pressure).mean()))

        if (t % plot_interval == 0) or (t == steps - 1):
            ax_field.clear()
            ax_field.plot(xs, (truth + 1) / 2, "k--", alpha=0.3, label="Truth")
            ax_field.plot(xs, beliefs, "b-", linewidth=2, label="Beliefs u(x)")
            dom_size = n_points // n_domains
            for d in range(1, n_domains):
                ax_field.axvline(
                    d * dom_size,
                    color="red",
                    alpha=0.3,
                    linestyle=":",
                    linewidth=1,
                )
            ax_field.set_ylim(-0.1, 1.1)
            ax_field.set_ylabel("u(x)")
            ax_field.set_title(f"Belief Field at step {t}")
            ax_field.legend()
            ax_field.grid(alpha=0.3)

            ax_diff.clear()
            ax_diff.plot(xs, D, linewidth=2)
            ax_diff.set_ylabel("D(x)")
            ax_diff.set_xlabel("Position")
            ax_diff.set_title("Heterogeneous Diffusion")
            ax_diff.grid(alpha=0.3)

            ax_adv.clear()
            ax_adv.plot(xs, np.abs(adv_pressure), "r-", linewidth=2)
            ax_adv.set_ylabel("|Adversarial Pressure|")
            ax_adv.set_xlabel("Position")
            ax_adv.set_title(f"Adversarial Attack Pattern (t={t})")
            ax_adv.grid(alpha=0.3)

            ax_metrics.clear()
            ax_metrics.plot(acc_hist, label="Accuracy", linewidth=2)
            ax_metrics.plot(coh_hist, label="Global Coherence", linewidth=2)
            ax_metrics.plot(lam_hist, label="Mean λ_eff", linewidth=2)
            ax_metrics.plot(
                np.array(adv_hist) * 10,
                label="Adv. Strength (×10)",
                linewidth=2,
                linestyle="--",
                alpha=0.7,
            )
            ax_metrics.set_ylim(0.0, 1.1)
            ax_metrics.set_xlabel("Step")
            ax_metrics.set_ylabel("Metric")
            ax_metrics.legend(loc="best")
            ax_metrics.grid(alpha=0.3)

            plt.pause(0.001)

    plt.ioff()
    plt.show()

    print("\n" + "=" * 60)
    print("ALIGNMENT AS COHERENCE — Extended Summary")
    print("=" * 60)
    print(f"Final accuracy:           {acc_hist[-1]:.3f}")
    print(f"Final global coherence:   {coh_hist[-1]:.3f}")
    print(f"Mean D(x):                {D.mean():.3f}")
    print(f"D(x) range:               [{D.min():.3f}, {D.max():.3f}]")
    print(f"Number of domains:        {n_domains}")
    print(f"Adversarial strength:     {adversarial_strength:.3f}")
    print("\nKey observations:")
    print("- Domain boundaries act as coherence bottlenecks.")
    print("- Adversarial pressure adapts to target high-coherence regions.")
    print("- Global feedback helps maintain stability under attack.")
    print("- System exhibits resilience via spatially distributed coherence.")
    print("=" * 60)


if __name__ == "__main__":
    run_extended_simulation()
