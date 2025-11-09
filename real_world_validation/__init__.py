"""Real-world validation scaffolding for Alignment-as-Coherence."""

from .coherence_validation_template import bootstrap_example, describe_pipeline
from .load_sae_features_stub import SAEFeatureBundle, load_sae_features
from .map_to_fields_stub import CoherenceFieldSet, map_features_to_fields
from .run_real_model_validation import run_real_model_validation

__all__ = [
    "SAEFeatureBundle",
    "CoherenceFieldSet",
    "load_sae_features",
    "map_features_to_fields",
    "run_real_model_validation",
    "bootstrap_example",
    "describe_pipeline",
]
