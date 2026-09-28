"""
Names of every key AtlasRoute writes to an AnnData object, plus file-format constants.

Analysis code should address AnnData slots through these constants rather than
string literals so a rename happens in one place.
"""

from __future__ import annotations

PREFIX = "atlasroute"

# obs columns written by transport() (v0.2) and by ValidationResult.to_obs()
OBS_STATE = "atlasroute_state"
OBS_CONFIDENCE = "atlasroute_confidence"
OBS_UNKNOWN = "atlasroute_unknown"
OBS_MARGIN = "atlasroute_margin"
OBS_SECOND_BEST = "atlasroute_second_best"
OBS_KNN_SUPPORT = "atlasroute_knn_support"
OBS_NOVELTY = "atlasroute_novelty"  # reserved for novelty.py
OBS_LODO_STATE = "atlasroute_lodo_state"  # held-out prediction written back to the reference

# obsm keys
OBSM_PCA = "X_atlasroute_pca"  # PCA fallback destination when no latent key is given
OBSM_PROBA = "atlasroute_proba"  # reserved for transport(): cells x states probabilities

# uns keys
UNS_REFERENCE = "atlasroute"  # provenance dict written by AtlasReference.fit()
UNS_VALIDATION = "atlasroute_validation"

# labels
UNKNOWN_LABEL = "unknown"
STABILITY_CATEGORIES = ("stable", "moderate", "fragile", "insufficient_donors")
VALIDATION_METHODS = ("leave_one_donor_out", "random_split")

# frozen reference format
FORMAT_VERSION = 1
FROZEN_SUFFIX = ".atlasroute"
MANIFEST_FILE = "manifest.json"
