"""Projection utilities mapping feature activations into coherence fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np

from .load_sae_features_stub import SAEFeatureBundle


@dataclass
class CoherenceFieldSet:
    """Collection of 1D fields required by the diagnostics."""

    x_axis: np.ndarray
    v_deep: np.ndarray
    v_surface: np.ndarray
    v_target: np.ndarray
    normalization: Literal["zscore", "minmax", "none"] = "zscore"

    def validate_monotonic_axis(self) -> None:
        """Check that the x-axis is strictly increasing for spatial operators."""

        if np.any(np.diff(self.x_axis) <= 0):
            raise ValueError(
                "The coherence field axis must be strictly increasing. "
                "Consider sorting by feature salience or positional index."
            )

    def ensure_unit_shape(self) -> None:
        """Guarantee that the three fields share identical length."""

        field_lengths = {len(self.v_deep), len(self.v_surface), len(self.v_target)}
        if len(field_lengths) != 1:
            raise ValueError(
                "All coherence fields must have identical lengths. "
                f"Observed lengths: {field_lengths}"
            )

    def to_serializable(self) -> Dict[str, object]:
        """Return a JSON-friendly dictionary representation of the fields."""

        return {
            "x_axis": self.x_axis.tolist(),
            "v_deep": self.v_deep.tolist(),
            "v_surface": self.v_surface.tolist(),
            "v_target": self.v_target.tolist(),
            "normalization": self.normalization,
        }


def _collapse_feature_tensor(tensor: np.ndarray, *, length: int) -> np.ndarray:
    """Reduce an activation tensor to a 1D summary per feature index."""

    array = np.asarray(tensor, dtype=np.float64)
    if array.ndim == 1:
        if array.shape[0] != length:
            raise ValueError(
                "1D activation vector does not match the feature index length."
            )
        return array

    if array.size % length != 0:
        raise ValueError(
            "Activation tensor cannot be evenly reshaped using the feature index "
            "length. Provide data with consistent leading dimension."
        )

    reshaped = array.reshape(length, -1)
    return reshaped.mean(axis=1)


def map_features_to_fields(features: SAEFeatureBundle) -> CoherenceFieldSet:
    """Map high-dimensional features to the 1D fields expected by the simulators."""

    feature_index = np.asarray(features.feature_index, dtype=np.float64).ravel()
    if feature_index.size == 0:
        raise ValueError("Feature index is empty; cannot construct coherence fields.")

    order = np.argsort(feature_index)
    sorted_index = feature_index[order]

    index_min = float(sorted_index[0])
    index_max = float(sorted_index[-1])
    if np.isclose(index_min, index_max):
        x_axis = np.linspace(0.0, 1.0, sorted_index.size, dtype=np.float64)
    else:
        x_axis = (sorted_index - index_min) / (index_max - index_min)

    v_surface = _collapse_feature_tensor(
        features.pre_training_activations, length=feature_index.size
    )[order]
    v_deep = _collapse_feature_tensor(
        features.post_training_activations, length=feature_index.size
    )[order]
    v_target = _collapse_feature_tensor(
        features.target_alignment_activations, length=feature_index.size
    )[order]

    fields = CoherenceFieldSet(
        x_axis=x_axis,
        v_deep=v_deep,
        v_surface=v_surface,
        v_target=v_target,
        normalization="zscore",
    )
    fields.ensure_unit_shape()
    fields.validate_monotonic_axis()
    return fields


__all__ = ["CoherenceFieldSet", "map_features_to_fields"]
