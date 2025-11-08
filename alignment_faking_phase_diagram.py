#!/usr/bin/env python3
"""
alignment_faking_phase_diagram.py

Advanced Coherence-Based Alignment Analysis

Extends the alignment_faking_detector with:

1. Critical Training Intensity Finder
2. Temporal Island Evolution Tracker
3. Phase Diagram Generator (training_strength × coupling_to_deep)
4. Lyapunov Stability Analysis
5. Multi-scale Coherence Analysis

Goal:
Not just detect faking, but map when it disappears:
estimate the minimum training needed for genuine alignment and
characterize regimes of deceptive vs stable behavior.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from dataclasses import dataclass


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
    """One training update for surface values."""
    if rng is None:
        rng = np.random.default_rng()

    diff_term = D_surface * laplacian_1d(v_surface, dx)
    deep_term = coupling_to_deep * (v_deep - v_surface)
    train_term = training_strength * (v_target - v_surface)
    noise_term = noise_scale * rng.standard_normal(len(v_surface))

    dv = diff_term + deep_term + train_term + noise_term
    v_next = np.clip(v_surface + dt * dv, -1.0, 1.0)
    return v_next


@dataclass
class IslandData:
    start: int
    end: int
    size: int
    mean_mismatch: float
    max_mismatch: float


def detect_hidden_islands(
    v_deep: np.ndarray,
    v_surface: np.ndarray,
    agreement_threshold: float = 0.4,
    min_island_size: int = 5,
) -> Tuple[List[IslandData], np.ndarray]:
    """Detailed island detection: where |v_deep - v_surface| > threshold."""
    mismatch = np.abs(v_deep - v_surface)
    mask = mismatch > agreement_threshold

    islands: List[IslandData] = []
    in_island = False
    start = 0

    for i, active in enumerate(mask):
        if active and not in_island:
            in_island = True
            start = i
        elif not active and in_island:
            end = i
            if end - start >= min_island_size:
                region = mismatch[start:end]
                islands.append(IslandData(
                    start=start,
                    end=end,
                    size=end - start,
                    mean_mismatch=float(region.mean()),
                    max_mismatch=float(region.max()),
                ))
            in_island = False

    if in_island:
        end = len(mask)
        if end - start >= min_island_size:
            region = mismatch[start:end]
            islands.append(IslandData(
                start=start,
                end=end,
                size=end - start,
                mean_mismatch=float(region.mean()),
                max_mismatch=float(region.max()),
            ))

    return islands, mismatch


@dataclass
class TemporalSnapshot:
    step: int
    islands: List[IslandData]
    total_island_size: int
    mean_mismatch: float
    alignment_score: float


def track_island_evolution(
    v_deep: np.ndarray,
    v_target: np.ndarray,
    training_strength: float,
    train_steps: int = 400,
    snapshot_interval: int = 20,
    dx: float = 1.0,
    dt: float = 0.05,
    seed: int = 42,
) -> List[TemporalSnapshot]:
    """Track how misalignment islands evolve during training."""
    rng = np.random.default_rng(seed)
    v_surface = 0.5 * v_deep + 0.5 * v_target
    v_surface += 0.2 * rng.standard_normal(len(v_deep))
    v_surface = np.clip(v_surface, -1.0, 1.0)

    snapshots: List[TemporalSnapshot] = []

    for step in range(train_steps + 1):
        if step % snapshot_interval == 0:
            islands, mismatch = detect_hidden_islands(v_deep, v_surface)
            total_size = sum(i.size for i in islands)
            alignment_score = 1.0 - np.mean(np.abs(v_surface - v_target))
            snapshots.append(TemporalSnapshot(
                step=step,
                islands=islands,
                total_island_size=total_size,
                mean_mismatch=float(mismatch.mean()),
                alignment_score=float(alignment_score),
            ))

        if step < train_steps:
            v_surface = update_surface_values(
                v_surface, v_deep, v_target,
                dx, dt,
                training_strength=training_strength,
                rng=rng,
            )

    return snapshots


def analyze_island_persistence(
    snapshots: List[TemporalSnapshot],
) -> Dict:
    """Analyze persistence of misalignment islands over time."""
    if not snapshots:
        return {}

    sizes = [s.total_island_size for s in snapshots]
    mismatches = [s.mean_mismatch for s in snapshots]

    final_size = sizes[-1]
    max_size = max(sizes) if sizes else 0
    persistence = final_size / max_size if max_size > 0 else 0.0

    if len(sizes) > 1:
        decay_rate = (sizes[0] - sizes[-1]) / len(sizes)
    else:
        decay_rate = 0.0

    return {
        "persistence": float(persistence),
        "decay_rate": float(decay_rate),
        "final_size": int(final_size),
        "trajectory": sizes,
        "mismatch_trajectory": mismatches,
    }


def find_critical_training_strength(
    v_deep: np.ndarray,
    v_target: np.ndarray,
    strength_range: Tuple[float, float] = (0.1, 2.5),
    n_trials: int = 15,
    train_steps: int = 300,
    dx: float = 1.0,
    dt: float = 0.05,
    seed: int = 42,
) -> Dict:
    """Scan training_strength and find critical point where islands vanish."""
    strengths = np.linspace(*strength_range, n_trials)
    results: Dict[str, list] = {
        "strengths": strengths,
        "island_counts": [],
        "total_sizes": [],
        "mean_mismatches": [],
        "alignment_scores": [],
    }

    print("\nScanning training strength phase space...")
    print(f"{'Strength':<12} {'Islands':<10} {'Total Size':<12} {'Mismatch':<12} {'Status':<15}")
    print("-" * 65)

    for strength in strengths:
        rng = np.random.default_rng(seed)
        v_surface = 0.5 * v_deep + 0.5 * v_target
        v_surface += 0.2 * rng.standard_normal(len(v_deep))
        v_surface = np.clip(v_surface, -1.0, 1.0)

        for _ in range(train_steps):
            v_surface = update_surface_values(
                v_surface, v_deep, v_target,
                dx, dt,
                training_strength=strength,
                rng=rng,
            )

        islands, mismatch = detect_hidden_islands(v_deep, v_surface)
        total_size = sum(i.size for i in islands)
        mean_mismatch = float(mismatch.mean())
        alignment = 1.0 - np.mean(np.abs(v_surface - v_target))

        results["island_counts"].append(len(islands))
        results["total_sizes"].append(total_size)
        results["mean_mismatches"].append(mean_mismatch)
        results["alignment_scores"].append(float(alignment))

        if total_size > 50:
            status = "🔴 FAKING"
        elif total_size > 20:
            status = "🟡 TRANSITION"
        else:
            status = "🟢 GENUINE"

        print(f"{strength:<12.3f} {len(islands):<10} {total_size:<12} {mean_mismatch:<12.3f} {status:<15}")

    sizes = np.array(results["total_sizes"])
    if len(sizes) > 1:
        gradients = -np.diff(sizes)
        critical_idx = int(np.argmax(gradients))
        critical_strength = strengths[critical_idx]
    else:
        critical_strength = strength_range[0]

    results["critical_strength"] = float(critical_strength)

    print(f"\n🎯 Critical training strength: {critical_strength:.3f}")
    print("   Below this: hidden prefs persist / faking likely")
    print("   Above this: genuine internalization more plausible")

    return results


def compute_lyapunov_exponent(
    v_surface: np.ndarray,
    v_deep: np.ndarray,
    v_target: np.ndarray,
    training_strength: float,
    n_steps: int = 100,
    dx: float = 1.0,
    dt: float = 0.05,
    epsilon: float = 1e-4,
    seed: int = 42,
) -> float:
    """Toy Lyapunov exponent: positive = unstable, negative = stable."""
    rng = np.random.default_rng(seed)

    v1 = v_surface.copy()
    v2 = v_surface + epsilon * rng.standard_normal(len(v_surface))
    v2 = np.clip(v2, -1.0, 1.0)

    divergence_sum = 0.0
    valid_steps = 0

    for _ in range(n_steps):
        v1 = update_surface_values(
            v1, v_deep, v_target,
            dx, dt,
            training_strength=training_strength,
            noise_scale=0.01,
            rng=rng,
        )
        v2 = update_surface_values(
            v2, v_deep, v_target,
            dx, dt,
            training_strength=training_strength,
            noise_scale=0.01,
            rng=np.random.default_rng(seed + 1),
        )

        distance = np.linalg.norm(v1 - v2)
        if distance > epsilon * 0.1:
            divergence_sum += np.log(distance / epsilon)
            valid_steps += 1

            if distance > epsilon * 10:
                direction = v2 - v1
                norm = np.linalg.norm(direction) + 1e-12
                v2 = v1 + epsilon * direction / norm
                v2 = np.clip(v2, -1.0, 1.0)

    if valid_steps > 0:
        return float(divergence_sum / valid_steps)
    return 0.0


def analyze_multiscale_structure(
    v_deep: np.ndarray,
    v_surface: np.ndarray,
) -> Dict:
    """FFT-based multiscale analysis of mismatch."""
    mismatch = v_deep - v_surface
    n = len(mismatch)

    fft = np.fft.fft(mismatch)
    power = np.abs(fft[: n // 2]) ** 2
    freqs = np.fft.fftfreq(n)[: n // 2]

    if len(power) > 1:
        idx = int(np.argmax(power[1:]) + 1)
        dominant_freq = float(freqs[idx])
        dominant_power = float(power[idx])
    else:
        dominant_freq = 0.0
        dominant_power = 0.0

    total_power = float(power.sum())
    structure_score = dominant_power / total_power if total_power > 0 else 0.0

    return {
        "power_spectrum": power,
        "freqs": freqs,
        "dominant_freq": dominant_freq,
        "structure_score": float(structure_score),
    }


def generate_phase_diagram(
    v_deep: np.ndarray,
    v_target: np.ndarray,
    strength_range: Tuple[float, float] = (0.1, 2.5),
    coupling_range: Tuple[float, float] = (0.1, 0.8),
    n_points: int = 12,
    train_steps: int = 300,
    seed: int = 42,
) -> Dict:
    """2D phase diagram of faking intensity over (strength, coupling)."""
    strengths = np.linspace(*strength_range, n_points)
    couplings = np.linspace(*coupling_range, n_points)

    phase_map = np.zeros((n_points, n_points))

    print("\nGenerating phase diagram...")
    print(f"Sampling {n_points}x{n_points} = {n_points ** 2} parameter combinations")

    for i, coupling in enumerate(couplings):
        for j, strength in enumerate(strengths):
            rng = np.random.default_rng(seed)
            v_surface = 0.5 * v_deep + 0.5 * v_target
            v_surface += 0.2 * rng.standard_normal(len(v_deep))
            v_surface = np.clip(v_surface, -1.0, 1.0)

            for _ in range(train_steps):
                v_surface = update_surface_values(
                    v_surface, v_deep, v_target,
                    dx=1.0,
                    dt=0.05,
                    training_strength=strength,
                    coupling_to_deep=coupling,
                    rng=rng,
                )

            islands, _ = detect_hidden_islands(v_deep, v_surface)
            total_size = sum(i.size for i in islands)
            phase_map[i, j] = min(total_size / 100.0, 1.0)

        print(f"  Progress: {(i + 1) / n_points * 100:.0f}%")

    return {
        "phase_map": phase_map,
        "strengths": strengths,
        "couplings": couplings,
    }


def visualize_complete_analysis(
    xs: np.ndarray,
    v_deep: np.ndarray,
    v_target: np.ndarray,
    v_surface: np.ndarray,
    temporal_snapshots: List[TemporalSnapshot],
    phase_results: Dict,
    phase_diagram: Dict,
    lyapunov: float,
):
    """Multi-panel visualization of fields, islands, phase transition, phase diagram."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    islands, mismatch = detect_hidden_islands(v_deep, v_surface)

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(xs, v_deep, "g-", label="Deep Values", linewidth=2.5, alpha=0.8)
    ax1.plot(xs, v_target, "r--", label="Training Target", linewidth=2, alpha=0.6)
    ax1.plot(xs, v_surface, "b-", label="Surface Values", linewidth=2.5)
    for isl in islands:
        ax1.axvspan(xs[isl.start], xs[isl.end - 1], color="red", alpha=0.15)
    ax1.set_ylabel("Value")
    ax1.set_title("Value Fields & Hidden Preference Islands", fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(xs, mismatch, color="purple", linewidth=2)
    ax2.axhline(0.4, linestyle="--", color="red", alpha=0.5, label="Threshold")
    ax2.fill_between(xs, 0, mismatch, where=(mismatch > 0.4),
                     alpha=0.2, color="red")
    ax2.set_xlabel("Position")
    ax2.set_ylabel("|Deep - Surface|")
    ax2.set_title("Mismatch Profile", fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    steps = [s.step for s in temporal_snapshots]
    sizes = [s.total_island_size for s in temporal_snapshots]
    ax3.plot(steps, sizes, "ro-", linewidth=2, markersize=5)
    ax3.set_xlabel("Training Step")
    ax3.set_ylabel("Total Island Size")
    ax3.set_title("Island Persistence Over Time", fontweight="bold")
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    strengths = phase_results["strengths"]
    total_sizes = phase_results["total_sizes"]
    crit = phase_results["critical_strength"]
    ax4.plot(strengths, total_sizes, "bo-", linewidth=2, markersize=5)
    ax4.axvline(crit, color="red", linestyle="--", linewidth=2,
                label=f"Critical = {crit:.2f}")
    ax4.set_xlabel("Training Strength")
    ax4.set_ylabel("Island Size")
    ax4.set_title("Phase Transition in Faking", fontweight="bold")
    ax4.legend()
    ax4.grid(alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(strengths, phase_results["alignment_scores"],
             "go-", linewidth=2, markersize=5)
    ax5.axvline(crit, color="red", linestyle="--", linewidth=2, alpha=0.7)
    ax5.set_xlabel("Training Strength")
    ax5.set_ylabel("Alignment Score")
    ax5.set_title("Surface Alignment vs Strength", fontweight="bold")
    ax5.grid(alpha=0.3)

    ax6 = fig.add_subplot(gs[2, :])
    pm = phase_diagram["phase_map"]
    s_vals = phase_diagram["strengths"]
    c_vals = phase_diagram["couplings"]
    im = ax6.imshow(
        pm,
        aspect="auto",
        origin="lower",
        extent=[s_vals[0], s_vals[-1], c_vals[0], c_vals[-1]],
        cmap="RdYlGn_r",
    )
    ax6.set_xlabel("Training Strength")
    ax6.set_ylabel("Coupling to Deep")
    ax6.set_title("Phase Diagram: Faking Intensity", fontweight="bold")
    cbar = plt.colorbar(im, ax=ax6)
    cbar.set_label("Faking Intensity")

    S, C = np.meshgrid(s_vals, c_vals)
    CS = ax6.contour(S, C, pm, levels=[0.3, 0.6],
                     colors=["yellow", "red"], linewidths=2)
    ax6.clabel(CS, inline=True, fontsize=8)

    mid_c = 0.5 * (c_vals[0] + c_vals[-1])
    ax6.plot(crit, mid_c, "w*", markersize=16,
             markeredgecolor="black", markeredgewidth=1.2,
             label="Critical Line")
    ax6.legend(loc="upper right")

    fig.suptitle(
        f"Alignment Faking Phase Analysis | Lyapunov: {lyapunov:.3f} | "
        f"Islands: {len(islands)} | Critical Strength: {crit:.2f}",
        fontweight="bold",
    )

    plt.tight_layout()
    plt.show()


def run_complete_analysis(
    n_points: int = 400,
    train_steps: int = 400,
    seed: int = 7,
):
    """Run the full analysis pipeline."""
    print("=" * 70)
    print("ALIGNMENT FAKING PHASE ANALYSIS")
    print("Coherence-Based Detection with Critical Point Mapping")
    print("=" * 70)

    xs, v_deep, v_target, _ = init_value_fields(n_points, seed)

    print("\n[1/5] Finding critical training strength...")
    phase_results = find_critical_training_strength(
        v_deep, v_target,
        strength_range=(0.1, 2.5),
        n_trials=15,
        train_steps=300,
        dx=1.0,
        dt=0.05,
        seed=seed,
    )

    print("\n[2/5] Tracking island evolution at critical strength...")
    temporal_snapshots = track_island_evolution(
        v_deep, v_target,
        training_strength=phase_results["critical_strength"],
        train_steps=train_steps,
        snapshot_interval=20,
        dx=1.0,
        dt=0.05,
        seed=seed,
    )
    persistence_data = analyze_island_persistence(temporal_snapshots)
    print(f"   Island persistence score: {persistence_data['persistence']:.3f}")
    print(f"   Decay rate:               {persistence_data['decay_rate']:.3f}")

    print("\n[3/5] Generating 2D phase diagram...")
    phase_diagram = generate_phase_diagram(
        v_deep, v_target,
        strength_range=(0.1, 2.5),
        coupling_range=(0.1, 0.8),
        n_points=12,
        train_steps=250,
        seed=seed,
    )

    print("\n[4/5] Computing Lyapunov stability...")
    rng = np.random.default_rng(seed)
    v_surface_test = 0.5 * v_deep + 0.5 * v_target
    v_surface_test += 0.2 * rng.standard_normal(n_points)
    v_surface_test = np.clip(v_surface_test, -1.0, 1.0)

    lyapunov = compute_lyapunov_exponent(
        v_surface_test, v_deep, v_target,
        training_strength=phase_results["critical_strength"],
        n_steps=80,
        dx=1.0,
        dt=0.05,
        epsilon=1e-4,
        seed=seed,
    )
    print(f"   Lyapunov exponent: {lyapunov:.4f}")
    if lyapunov > 0.01:
        print("   → System unstable (deep prefs may reassert).")
    elif lyapunov < -0.01:
        print("   → System stable (surface alignment internalized).")
    else:
        print("   → System near critical regime.")

    print("\n[5/5] Training final model for visualization...")
    rng = np.random.default_rng(seed)
    v_surface = 0.5 * v_deep + 0.5 * v_target
    v_surface += 0.2 * rng.standard_normal(n_points)
    v_surface = np.clip(v_surface, -1.0, 1.0)
    for _ in range(train_steps):
        v_surface = update_surface_values(
            v_surface, v_deep, v_target,
            dx=1.0,
            dt=0.05,
            training_strength=phase_results["critical_strength"],
            rng=rng,
        )

    multiscale = analyze_multiscale_structure(v_deep, v_surface)
    islands, _ = detect_hidden_islands(v_deep, v_surface)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Critical training strength:    {phase_results['critical_strength']:.3f}")
    print(f"Number of persistent islands:  {len(islands)}")
    print(f"Total island coverage:         {sum(i.size for i in islands)} / {n_points}")
    print(f"Lyapunov exponent:             {lyapunov:.4f}")
    print(f"Structure score (FFT):         {multiscale['structure_score']:.3f}")
    print(f"Island persistence:            {persistence_data['persistence']:.3f}")
    print("\n📊 KEY INSIGHTS:")
    if phase_results["critical_strength"] < 0.5:
        print("   • Easy to retrain: deep prefs weak.")
    elif phase_results["critical_strength"] < 1.0:
        print("   • Moderate regime: normal alignment difficulty.")
    else:
        print("   • High regime: strong hidden prefs, faking risk.")
    if len(islands) > 3:
        print("   • Multiple islands → heterogeneous internal landscape.")
    if lyapunov > 0.01:
        print("   • Positive Lyapunov → fragile alignment.")
    if multiscale["structure_score"] > 0.3:
        print("   • Structured mismatch → organized hidden patterns.")
    print("=" * 70)

    print("\n[Generating visualization...]")
    visualize_complete_analysis(
        xs, v_deep, v_target, v_surface,
        temporal_snapshots, phase_results, phase_diagram,
        lyapunov,
    )

    return {
        "phase_results": phase_results,
        "temporal_snapshots": temporal_snapshots,
        "phase_diagram": phase_diagram,
        "lyapunov": lyapunov,
        "multiscale": multiscale,
        "final_islands": islands,
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            print("Running quick analysis (reduced resolution)...")
            run_complete_analysis(n_points=200, train_steps=200, seed=7)
        elif sys.argv[1] == "--help":
            print("Usage: python alignment_faking_phase_diagram.py [--quick|--help]")
            print("  (none)   : Full analysis")
            print("  --quick  : Faster, lower-res phase sweep")
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        run_complete_analysis(n_points=400, train_steps=400, seed=7)
