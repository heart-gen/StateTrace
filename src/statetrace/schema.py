"""
Names of every key StateTrace writes to an AnnData object, category vocabularies and
file-format constants. Analysis code addresses AnnData slots through these constants.
"""

from __future__ import annotations

PREFIX = "statetrace"

# obs columns written by TransportRoute.to_obs() and ValidationResult.to_obs()
OBS_STATE = "statetrace_state"
OBS_CONFIDENCE = "statetrace_confidence"
OBS_UNCERTAINTY = "statetrace_uncertainty"
OBS_MARGIN = "statetrace_margin"
OBS_SECOND_BEST = "statetrace_second_best"
OBS_SUPPORT = "statetrace_support"  # local reference support (neighbour vote fraction)
OBS_NOVELTY = "statetrace_novelty"  # reference-distance z-score
OBS_CATEGORY = "statetrace_category"  # supported / uncertain / out_of_reference
OBS_HOLDOUT_STATE = "statetrace_holdout_state"  # held-out prediction written back to the reference

# obsm / uns keys
OBSM_PCA = "X_statetrace_pca"
OBSM_PROBA = "statetrace_proba"
UNS_REFERENCE = "statetrace"
UNS_VALIDATION = "statetrace_validation"
UNS_TRANSPORT = "statetrace_transport"

# vocabularies
UNKNOWN_LABEL = "unknown"
STABILITY_CATEGORIES = ("stable", "moderate", "fragile", "insufficient")
CELL_CATEGORIES = ("supported", "uncertain", "out_of_reference")
TRANSPORT_STATUS = ("supported", "supported_with_caution", "shifted", "insufficient", "absent")
SHIFT_LEVELS = ("low", "moderate", "high", "unknown")
VALIDATION_METHODS = ("leave_one_group_out", "leave_one_study_out", "random_split")

# frozen reference format
FORMAT_VERSION = 2
FROZEN_SUFFIX = ".statetrace"
MANIFEST_FILE = "manifest.json"
