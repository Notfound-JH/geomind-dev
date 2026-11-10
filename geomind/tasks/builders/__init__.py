from .base import BaseFamilyBuilder
from .binding import BindingBuilder
from .geometric import GeometricBuilder
from .reindex import ReindexBuilder
from .relation import RelationBuilder
from .track import TrackBuilder

FAMILY_BUILDERS = {
    "relation": RelationBuilder(),
    "spatial_placement": RelationBuilder(),
    "track": TrackBuilder(),
    "historical_identity_tracking": TrackBuilder(),
    "reindex": ReindexBuilder(),
    "state_reindexing": ReindexBuilder(),
    "geometric": GeometricBuilder(),
    "shape_completion": GeometricBuilder(),
    "binding": BindingBuilder(),
    "anchor_reference": BindingBuilder(),
}


def resolve_builder(family: str) -> BaseFamilyBuilder:
    return FAMILY_BUILDERS.get(str(family).strip().lower(), BaseFamilyBuilder())
