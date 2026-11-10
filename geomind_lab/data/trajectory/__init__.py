"""Reusable trajectory collection, annotation, resampling, and HDF5 export APIs."""

from .augmentation import IncrementPlanConfig, WaypointPlan, sample_increment_candidates
from .collector import EpisodeCollector, StepRecord
from .hdf5_exporter import CameraH5Stream, export_episode_h5
from .metadata import build_trajectory_metadata_contract, configured_action_scale
from .resample import resample_episode
from .segmenter import (
    ATOMIC_ACTION_NAMES,
    SEGMENTER_VERSION_EVENT_KINEMATICS_V2,
    SEGMENTER_VERSION_STAGE_V1,
    SegmentationConfig,
    SegmentationResult,
    build_segments,
    segment_records,
    stage_to_atomic_id,
)
from .validator import validate_dataset_manifest, validate_episode_h5

__all__ = [
    "ATOMIC_ACTION_NAMES",
    "SEGMENTER_VERSION_EVENT_KINEMATICS_V2",
    "SEGMENTER_VERSION_STAGE_V1",
    "CameraH5Stream",
    "EpisodeCollector",
    "IncrementPlanConfig",
    "SegmentationConfig",
    "SegmentationResult",
    "StepRecord",
    "WaypointPlan",
    "build_trajectory_metadata_contract",
    "build_segments",
    "configured_action_scale",
    "export_episode_h5",
    "resample_episode",
    "sample_increment_candidates",
    "segment_records",
    "stage_to_atomic_id",
    "validate_dataset_manifest",
    "validate_episode_h5",
]
