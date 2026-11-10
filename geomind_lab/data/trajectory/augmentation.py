"""Seeded free-space 3D increment trajectory augmentation primitives."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
import random


XYZ = tuple[float, float, float]


@dataclass(frozen=True)
class IncrementPlanConfig:
	segment_count: int = 6
	candidate_count: int = 16
	reverse_probability: float = 0.75
	reverse_ratio: float = 0.30
	max_reverse_absolute_m: float = 0.08
	zero_axis_bound_m: float = 0.05
	max_segment_length_m: float = 0.14
	max_segment_ratio: float = 0.40


@dataclass(frozen=True)
class WaypointPlan:
	waypoints: tuple[XYZ, ...]
	increment_vectors: tuple[XYZ, ...] = ()
	increment_permutation: tuple[int, ...] = ()
	path_style: str = "canonical"
	augmented: bool = False
	fallback_reason: str = ""


@dataclass(frozen=True)
class PlanMetrics:
	path_length_m: float
	max_lateral_deviation_m: float
	max_segment_length_m: float
	endpoint_error_m: float
	axis_signature: tuple[str, ...]


@dataclass(frozen=True)
class CandidateEvaluation:
	candidate_index: int
	plan: WaypointPlan
	rejection_reasons: tuple[str, ...] = ()
	metrics: PlanMetrics | None = None

	@property
	def accepted(self) -> bool:
		return not self.rejection_reasons


CandidateValidator = Callable[[WaypointPlan], Sequence[str]]


def _xyz(value: Sequence[float], name: str) -> XYZ:
	if len(value) != 3:
		raise ValueError(f"{name} must contain exactly three values.")
	return (float(value[0]), float(value[1]), float(value[2]))


def _weights(rng: random.Random, count: int) -> list[float]:
	values = [rng.expovariate(1.0) for _ in range(count)]
	total = sum(values)
	return [value / total for value in values]


def _axis_increments(rng: random.Random, delta: float, config: IncrementPlanConfig) -> list[float]:
	count = config.segment_count
	if abs(delta) <= 1e-9:
		raw = [rng.uniform(-config.zero_axis_bound_m, config.zero_axis_bound_m) for _ in range(count)]
		centered = [value - sum(raw) / count for value in raw]
		scale = max((abs(value) for value in centered), default=0.0)
		strict_bound = math.nextafter(config.zero_axis_bound_m, 0.0)
		return centered if scale < strict_bound or scale <= 1e-9 else [value * strict_bound / scale for value in centered]

	direction = 1.0 if delta > 0.0 else -1.0
	reverse = 0.0
	if count > 1 and rng.random() < config.reverse_probability:
		reverse = min(
		abs(delta) * config.reverse_ratio,
		config.max_reverse_absolute_m,
		math.nextafter(config.zero_axis_bound_m, 0.0),
	)
	if reverse <= 1e-9:
		return [direction * abs(delta) * weight for weight in _weights(rng, count)]

	reverse_index = rng.randrange(count)
	positive_weights = iter(_weights(rng, count - 1))
	return [
		-direction * reverse if index == reverse_index else direction * (abs(delta) + reverse) * next(positive_weights)
		for index in range(count)
	]


def _distance(left: XYZ, right: XYZ) -> float:
	return math.sqrt(sum((right[axis] - left[axis]) ** 2 for axis in range(3)))


def _axis_signature(increments: Sequence[XYZ]) -> tuple[str, ...]:
	result: list[str] = []
	for increment in increments:
		axis = max(range(3), key=lambda index: abs(increment[index]))
		result.append(("+" if increment[axis] >= 0.0 else "-") + "xyz"[axis])
	return tuple(result)


def increment_plan_metrics(plan: WaypointPlan) -> PlanMetrics:
	if len(plan.waypoints) < 2:
		return PlanMetrics(0.0, 0.0, 0.0, 0.0, ())
	start, end = plan.waypoints[0], plan.waypoints[-1]
	direct = tuple(end[axis] - start[axis] for axis in range(3))
	direct_sq = sum(value * value for value in direct)
	segments = tuple(
		tuple(right[axis] - left[axis] for axis in range(3))
		for left, right in zip(plan.waypoints, plan.waypoints[1:])
	)
	lengths = [_distance(left, right) for left, right in zip(plan.waypoints, plan.waypoints[1:])]
	deviations: list[float] = []
	for point in plan.waypoints[1:-1]:
		if direct_sq <= 1e-12:
			projection = start
		else:
			ratio = sum((point[axis] - start[axis]) * direct[axis] for axis in range(3)) / direct_sq
			projection = tuple(start[axis] + ratio * direct[axis] for axis in range(3))
		deviations.append(_distance(point, projection))
	integrated = tuple(sum(segment[axis] for segment in segments) for axis in range(3))
	endpoint_error = math.sqrt(sum((integrated[axis] - direct[axis]) ** 2 for axis in range(3)))
	return PlanMetrics(sum(lengths), max(deviations, default=0.0), max(lengths, default=0.0), endpoint_error, _axis_signature(plan.increment_vectors or segments))


def interpolate_waypoint_plan(
	plan: WaypointPlan,
	*,
	max_speed_mps: float,
	step_dt: float,
) -> list[XYZ]:
	"""Densify waypoints with smoothstep while respecting a per-step speed budget."""
	max_step_m = max(float(max_speed_mps) * float(step_dt) / 1.875, 1e-4)
	targets: list[XYZ] = []
	for start, end in zip(plan.waypoints, plan.waypoints[1:]):
		distance = _distance(start, end)
		steps = max(1, int(math.ceil(distance / max_step_m)))
		for step_index in range(1, steps + 1):
			progress = float(step_index) / float(steps)
			alpha = progress * progress * (3.0 - 2.0 * progress)
			targets.append(tuple(start[axis] + alpha * (end[axis] - start[axis]) for axis in range(3)))
	return targets


def candidate_evaluations_payload(evaluations: Sequence[CandidateEvaluation]) -> list[dict[str, object]]:
	return [
		{
			"candidate_index": int(evaluation.candidate_index),
			"waypoints": [list(point) for point in evaluation.plan.waypoints],
			"accepted": bool(evaluation.accepted),
			"rejection_reasons": list(evaluation.rejection_reasons),
			"path_length_m": float(evaluation.metrics.path_length_m) if evaluation.metrics else 0.0,
			"max_lateral_deviation_m": float(evaluation.metrics.max_lateral_deviation_m) if evaluation.metrics else 0.0,
			"max_segment_length_m": float(evaluation.metrics.max_segment_length_m) if evaluation.metrics else 0.0,
			"endpoint_error_m": float(evaluation.metrics.endpoint_error_m) if evaluation.metrics else 0.0,
			"axis_signature": [] if evaluation.metrics is None else list(evaluation.metrics.axis_signature),
		}
		for evaluation in evaluations
	]


def sample_increment_candidates(
	start_xyz: Sequence[float],
	end_xyz: Sequence[float],
	*,
	seed: int,
	config: IncrementPlanConfig,
) -> tuple[WaypointPlan, ...]:
	"""Generate endpoint-preserving paths by shuffling complete 3D increments."""
	if config.segment_count < 2:
		raise ValueError("segment_count must be at least two.")
	if config.candidate_count < 0:
		raise ValueError("candidate_count must be non-negative.")
	start, end = _xyz(start_xyz, "start_xyz"), _xyz(end_xyz, "end_xyz")
	delta = tuple(end[axis] - start[axis] for axis in range(3))
	if config.candidate_count == 0 or max(abs(value) for value in delta) <= 1e-9:
		return ()
	rng = random.Random(seed)
	plans: list[WaypointPlan] = []
	for candidate_index in range(config.candidate_count):
		axes = [_axis_increments(rng, value, config) for value in delta]
		vectors = [(axes[0][index], axes[1][index], axes[2][index]) for index in range(config.segment_count)]
		permutation = list(range(config.segment_count))
		style_index = candidate_index % 3
		direction = 1.0 if delta[2] >= 0.0 else -1.0
		if style_index == 0:
			style = "vertical_first"
			permutation = sorted(permutation, key=lambda index: direction * vectors[index][2], reverse=True)
		elif style_index == 1:
			style = "lateral_first"
			permutation = sorted(
				permutation,
				key=lambda index: vectors[index][0] ** 2 + vectors[index][1] ** 2,
				reverse=True,
			)
		else:
			style = "mixed"
			rng.shuffle(permutation)
		increments = tuple(vectors[index] for index in permutation)
		points: list[XYZ] = [start]
		current = list(start)
		for increment in increments:
			current = [current[axis] + increment[axis] for axis in range(3)]
			points.append((current[0], current[1], current[2]))
		points[-1] = end
		plan = WaypointPlan(tuple(points), increments, tuple(permutation), style, True)
		plans.append(plan)
	return tuple(plans)


def choose_valid_waypoint_candidate(
	candidates: Sequence[WaypointPlan], *, validator: CandidateValidator, fallback: WaypointPlan,
	preferred_path_style: str | None = None,
) -> tuple[WaypointPlan, tuple[CandidateEvaluation, ...]]:
	"""Validate every candidate, then choose the most distinct valid path."""
	evaluations = tuple(
		CandidateEvaluation(index, candidate, tuple(str(reason) for reason in validator(candidate) if str(reason)), increment_plan_metrics(candidate))
		for index, candidate in enumerate(candidates)
	)
	accepted = [evaluation for evaluation in evaluations if evaluation.accepted]
	if accepted:
		selected = max(
			accepted,
			key=lambda evaluation: (
				int(preferred_path_style is not None and evaluation.plan.path_style == preferred_path_style),
				evaluation.metrics.max_lateral_deviation_m if evaluation.metrics else 0.0,
				-(evaluation.metrics.path_length_m if evaluation.metrics else 0.0),
				-evaluation.candidate_index,
			),
		)
		return selected.plan, evaluations
	return WaypointPlan(fallback.waypoints, fallback_reason="no_valid_increment_plan" if candidates else "no_increment_candidate_generated"), evaluations
