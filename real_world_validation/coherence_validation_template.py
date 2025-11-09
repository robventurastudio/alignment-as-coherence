"""Specialized template for real-world coherence validation workflows."""

from __future__ import annotations

from pprint import pprint
from typing import Optional

from .load_sae_features_stub import SAEFeatureBundle, load_sae_features
from .map_to_fields_stub import CoherenceFieldSet, map_features_to_fields
from .run_real_model_validation import run_real_model_validation


def describe_pipeline() -> str:
    """Return a human-readable summary of the validation workflow."""

    return (
        "1. load_sae_features -> 2. map_features_to_fields -> "
        "3. run_real_model_validation"
    )


def bootstrap_example(project_root: Optional[str] = None) -> None:
    """Guide practitioners through populating the scaffolding."""

    print("Alignment-as-Coherence real-world validation template")
    print("Pipeline:", describe_pipeline())
    print()
    print("Customize load_sae_features to ingest lab-specific activations.")
    print(
        "Customize map_features_to_fields to translate activations into "
        "coherence fields."
    )
    print(
        "run_real_model_validation now executes end-to-end using the "
        "synthetic fallback; replace the helpers to integrate real data."
    )
    if project_root is not None:
        print("Suggested project root:", project_root)

    print()
    print("--- Synthetic dry-run (override the helpers for real data) ---")
    result = run_real_model_validation(project_root)
    summary = {
        "bundle_source": result.get("bundle_metadata", {}).get("source", "synthetic"),
        "num_features": len(result.get("bundle_index", [])),
        "hidden_island_clusters": len(result.get("hidden_islands", [])),
    }
    pprint(summary)


__all__ = [
    "SAEFeatureBundle",
    "CoherenceFieldSet",
    "load_sae_features",
    "map_features_to_fields",
    "run_real_model_validation",
    "bootstrap_example",
    "describe_pipeline",
]
