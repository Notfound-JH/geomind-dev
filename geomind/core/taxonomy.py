"""Canonical public task families, independent of runtime oracle protocols."""

CANONICAL_TASK_FAMILIES = {
    "TargetIdentification": (
        "SelectByOrder", "SelectByGridIndex", "SelectByBranchPosition", "SelectByRegion",
    ),
    "RelationalGrounding": (
        "SelectByDirection", "SelectByDistance", "SelectByAdjacency",
        "SelectByPathDistance", "PlaceByTopology",
    ),
    "ConstraintComposition": (
        "SelectWithinSubset", "SelectWithVisualCue", "SelectAfterExclusion",
        "SelectByTwoRelations", "SelectByIndirectReference",
    ),
    "GeometricInference": (
        "InferMidpoint", "InferPointSymmetry", "ReflectAcrossLine", "ExtendEqualSpacing",
        "ProjectOntoLine", "CompleteSquare", "ContinueSpatialPattern",
    ),
    "SceneUpdating": ("DetectSceneChange", "UpdateObjectOrder", "UpdateEmptySlot"),
    "IdentityTracking": ("TrackMovingObject", "TrackObjectSwap", "TrackObjectOrigin"),
}

TASK_TO_FAMILY = {
    task_id: family
    for family, task_ids in CANONICAL_TASK_FAMILIES.items()
    for task_id in task_ids
}

FAMILY_LABELS = {
    "TargetIdentification": "Target Identification",
    "RelationalGrounding": "Relational Grounding",
    "ConstraintComposition": "Constraint Composition",
    "GeometricInference": "Geometric Inference",
    "SceneUpdating": "Scene Updating",
    "IdentityTracking": "Identity Tracking",
}
