from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import gymnasium as gym
from geomind.episodes.resolver import load_json

from geomind_lab.mimic.utils.task_metadata_registry import (
    build_task_metadata,
    resolve_task_id as resolve_legacy_task_id,
    supported_task_versions,
)

from .ids import is_public_task_id, parse_task_id
from .types import TaskSpec
from .schema import normalize_condition, normalize_task_config

_SUPPORTED_GEOMIND_TASK_VERSIONS = ("base", "semantic", "visual", "execution", "vla")


class GeoMindRegistry:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parents[1]
        self.tasks_dir = self.root / "configs" / "tasks"
        self._task_specs: dict[str, TaskSpec] = {}
        self._aliases: dict[str, str] = {}
        self._payloads: dict[str, dict[str, Any]] = {}
        self._paths: dict[str, Path] = {}
        self._definitions: dict[str, dict[str, Any]] = {}
        self._names: dict[str, list[str]] = {}

    @staticmethod
    def _normalize_text(value: Any) -> str:
        return str(value).strip()

    @staticmethod
    def _normalize_task_version(value: Any) -> str:
        normalized = str(value).strip().lower()
        mapping = {
            "base": "base",
            "sem": "semantic",
            "semantic": "semantic",
            "visual": "visual",
            "execution": "execution",
            "exec": "execution",
            "vla": "vla",
        }
        return mapping.get(normalized, normalized)

    @staticmethod
    def _read_task_metadata_version_hint(payload: dict[str, Any]) -> str:
        meta_payload = payload.get("meta", {})
        if not isinstance(meta_payload, dict):
            return ""
        task_metadata = meta_payload.get("task_metadata", {})
        if not isinstance(task_metadata, dict):
            return ""
        return str(task_metadata.get("task_version", "")).strip()

    def _resolve_canonical_ids(self, *, task_id: str, payload: dict[str, Any]) -> tuple[str, str]:
        payload_canonical = self._normalize_text(payload.get("canonical_task_id", ""))
        payload_legacy = self._normalize_text(payload.get("legacy_task_id", ""))
        resolved_canonical = resolve_legacy_task_id(payload_canonical) or payload_canonical
        resolved_legacy = resolve_legacy_task_id(payload_legacy) or payload_legacy
        if resolved_canonical and resolved_legacy and resolved_canonical != resolved_legacy:
            raise ValueError(
                f"canonical_task_id/legacy_task_id mismatch for {task_id}: "
                f"{payload_canonical!r} vs {payload_legacy!r}"
            )
        canonical_task_id = resolved_canonical or resolved_legacy or resolve_legacy_task_id(task_id) or (payload_legacy or task_id)
        return canonical_task_id, (resolved_legacy or payload_legacy)

    def _require_metadata_match(
        self,
        *,
        task_id: str,
        field_name: str,
        payload_value: Any,
        metadata_value: Any,
    ) -> None:
        local_value = self._normalize_text(payload_value)
        canonical_value = self._normalize_text(metadata_value)
        if not local_value or not canonical_value:
            return
        if local_value.lower() != canonical_value.lower():
            raise ValueError(
                f"Config field {field_name!r} for {task_id} conflicts with legacy metadata: "
                f"local={local_value!r}, legacy={canonical_value!r}"
            )

    def _to_spec(self, payload: dict[str, Any]) -> TaskSpec:
        payload = normalize_task_config(payload)
        task_id = str(payload["task_id"])
        parsed = parse_task_id(task_id)
        status = payload.get("implementation_status", "implemented")
        if status not in {"implemented", "placeholder"}:
            raise ValueError(f"Unknown implementation_status for {task_id}: {status!r}")
        if status == "placeholder":
            return TaskSpec(
                task_id=task_id,
                canonical_task_id=str(payload.get("canonical_task_id", "")),
                family=str(payload["family"]).lower(),
                split=str(payload["split"]), variant=str(payload["variant"]),
                task_version=str(payload["task_version"]),
                task_name=str(payload["task_name"]), version=int(payload["version"]),
                scene_id=-1, instruction="", evaluator_key="",
                public_task_id=str(payload["public_task_id"]),
                task_family=str(payload["task_family"]),
                condition=str(payload["condition"]), layout=str(payload["layout"]),
                interface=str(payload["interface"]),
                task_variant=str(payload.get("public_variant", "")),
                parameters=deepcopy(payload.get("parameters", {})),
                implementation_status="placeholder",
            )
        if payload.get('backend')=='assembled_native':
            import importlib
            native=deepcopy(payload.get('native_assembly',{}))
            module_name=native.get('recipe_module','')
            if (payload.get('protocol_family')!='canonical_extension'
                    or module_name not in {
                        'geomind.episodes.extended_static_tasks',
                        'geomind.episodes.extended_temporal_tasks',
                        'geomind.episodes.extended_geometry_tasks'}):
                raise ValueError(f'Invalid native assembled recipe binding for {task_id}')
            module=importlib.import_module(module_name)
            if task_id not in module.TASK_IDS or not all(callable(getattr(module,key,None)) for key in ('build_spec','solve_spec')):
                raise ValueError(f'Native assembled recipe is unavailable for {task_id}')
            compiled=module.build_spec(task_id,scene_id=int(payload['scene_id']),seed=int(payload.get('backend_kwargs',{}).get('seed',0)))
            instruction=compiled.get('instruction','')
            if (payload.get('interface')!='Standard' or payload.get('condition')!='Base'
                    or payload.get('task_version')!='base' or payload.get('variant')!='base'
                    or native.get('registration_id')!=task_id
                    or compiled.get('protocol',{}).get('gt_spec',{}).get('resolver')!=payload.get('public_task_id')):
                raise ValueError(f'Native assembled registration identity or interface mismatch for {task_id}')
            if not instruction or not compiled.get('runtime_objects'):
                raise ValueError(f'Native assembled recipe has no instruction or physical objects for {task_id}')
            return TaskSpec(task_id=task_id,canonical_task_id=str(payload['public_task_id']),
                family='canonical_extension',split=str(payload['split']),variant=str(payload['variant']),
                task_version=str(payload['task_version']),task_name=str(payload['task_name']),version=int(payload['version']),
                scene_id=int(payload['scene_id']),instruction=instruction,evaluator_key='assembled_object_state',
                public_task_id=str(payload['public_task_id']),task_family=str(payload['task_family']),
                condition=str(payload['condition']),layout=str(payload['layout']),interface=str(payload['interface']),
                task_variant=str(payload.get('public_variant','')),parameters=deepcopy(payload.get('parameters',{})),
                semantic_module=module_name,skill_type=str(native.get('action','')),phase='native_assembled_manipulation',
                supported_task_versions=('base',),tags=('assembled_native','canonical_extension'),
                meta={'native_assembly':native},implementation_status='implemented',release_version='1.0')
        canonical_task_id, legacy_task_id = self._resolve_canonical_ids(task_id=task_id, payload=payload)
        supported_versions = tuple(supported_task_versions(canonical_task_id))
        if not supported_versions:
            raise ValueError(
                f"Missing canonical supported_task_versions for {task_id}: canonical_task_id={canonical_task_id!r}"
            )
        payload_task_version = self._normalize_task_version(payload.get("task_version", ""))
        payload_variant = self._normalize_task_version(payload.get("variant", ""))
        task_id_variant = self._normalize_task_version(parsed.variant) if is_public_task_id(task_id) else ""
        metadata_version_hint = self._normalize_task_version(self._read_task_metadata_version_hint(payload))
        inferred_candidates = [item for item in (payload_variant, task_id_variant, metadata_version_hint) if item]
        unique_inferred_candidates = tuple(dict.fromkeys(inferred_candidates))
        if payload_task_version:
            normalized_task_version = payload_task_version
        elif len(unique_inferred_candidates) == 1:
            normalized_task_version = unique_inferred_candidates[0]
        elif len(unique_inferred_candidates) == 0 and len(supported_versions) == 1:
            normalized_task_version = supported_versions[0]
        else:
            raise ValueError(
                "Missing explicit task_version and unable to infer a unique version for "
                f"{task_id}. candidates={list(unique_inferred_candidates)!r}, supported={list(supported_versions)!r}"
            )
        if normalized_task_version not in _SUPPORTED_GEOMIND_TASK_VERSIONS:
            raise ValueError(
                f"Unsupported task_version token for {task_id}: {normalized_task_version!r}. "
                f"Known={list(_SUPPORTED_GEOMIND_TASK_VERSIONS)!r}"
            )
        if normalized_task_version not in supported_versions:
            raise ValueError(
                f"task_version for {task_id} is not supported by canonical metadata: "
                f"task_version={normalized_task_version!r}, supported={list(supported_versions)!r}"
            )
        if normalized_task_version != "vla":
            if payload_variant and payload_variant != normalized_task_version:
                raise ValueError(
                    f"variant/task_version mismatch for {task_id}: "
                    f"variant={payload_variant!r}, task_version={normalized_task_version!r}"
                )
            if task_id_variant and task_id_variant != normalized_task_version:
                raise ValueError(
                    f"task_id encoded variant mismatch for {task_id}: "
                    f"task_id_variant={task_id_variant!r}, task_version={normalized_task_version!r}"
                )
            if payload_variant and task_id_variant and payload_variant != task_id_variant:
                raise ValueError(
                    f"payload variant conflicts with task_id variant for {task_id}: "
                    f"payload_variant={payload_variant!r}, task_id_variant={task_id_variant!r}"
                )
        if metadata_version_hint and metadata_version_hint != normalized_task_version:
            raise ValueError(
                f"meta.task_metadata.task_version conflicts with normalized version for {task_id}: "
                f"meta_hint={metadata_version_hint!r}, task_version={normalized_task_version!r}"
            )
        metadata = build_task_metadata(canonical_task_id, task_version=normalized_task_version)
        metadata_task_version = self._normalize_task_version(metadata.get("task_version", ""))
        if metadata_task_version and metadata_task_version != normalized_task_version:
            raise ValueError(
                f"legacy metadata resolves different task_version for {task_id}: "
                f"metadata={metadata_task_version!r}, normalized={normalized_task_version!r}"
            )
        self._require_metadata_match(
            task_id=task_id,
            field_name="family",
            payload_value=payload.get("family", ""),
            metadata_value=metadata.get("family", ""),
        )
        self._require_metadata_match(
            task_id=task_id,
            field_name="evaluator_key",
            payload_value=payload.get("evaluator_key", ""),
            metadata_value=metadata.get("evaluator_key", ""),
        )
        self._require_metadata_match(
            task_id=task_id,
            field_name="task_version",
            payload_value=(payload_task_version or payload.get("task_version", "")),
            metadata_value=normalized_task_version,
        )
        if payload_task_version and payload_task_version != normalized_task_version:
            raise ValueError(
                f"Config field 'task_version' for {task_id} conflicts with canonical task version: "
                f"local={payload_task_version!r}, legacy={normalized_task_version!r}"
            )
        return TaskSpec(
            task_id=task_id,
            canonical_task_id=canonical_task_id,
            family=str(metadata.get("family") or payload.get("family") or parsed.family).lower(),
            split=str(payload.get("split", parsed.split)).lower(),
            variant=normalized_task_version,
            task_version=normalized_task_version,
            task_name=str(payload.get("task_name", parsed.task_name)),
            version=int(payload.get("version", parsed.version)),
            scene_id=int(payload["scene_id"]),
            instruction=str(payload.get("instruction", metadata.get("instruction", ""))),
            evaluator_key=str(metadata.get("evaluator_key") or payload.get("evaluator_key", "always_success")),
            public_task_id=str(payload.get("public_task_id", payload.get("task_name", parsed.task_name))),
            task_family=str(payload.get("task_family", payload.get("family", ""))),
            condition=str(payload.get("condition", "Base")),
            layout=str(payload.get("layout", "Custom")),
            interface=str(payload.get("interface", "Vla" if normalized_task_version == "vla" else "Standard")),
            task_variant=str(payload.get("public_variant", "")),
            parameters=deepcopy(payload.get("parameters", {})),
            semantic_module=str(metadata.get("semantic_module") or payload.get("semantic_module", "")),
            skill_type=str(metadata.get("skill_type") or payload.get("skill_type", "")),
            phase=str(metadata.get("phase") or payload.get("phase", "")),
            legacy_task_id=legacy_task_id,
            annotation_layout_key=str(payload.get("annotation_layout_key", "")),
            annotation_task_index=int(payload.get("annotation_task_index", 0)),
            object_set=str(payload.get("object_set", "")),
            assets_group=str(payload.get("assets_group", "")),
            supported_task_versions=supported_versions,
            tags=tuple(str(item) for item in payload.get("tags", [])),
            meta={
                **dict(payload.get("meta", {})),
                "task_metadata": metadata,
            },
        )

    def load(self) -> None:
        self._task_specs.clear()
        self._aliases.clear()
        self._payloads.clear()
        self._paths.clear()
        self._definitions.clear()
        self._names.clear()
        task_paths = sorted(self.tasks_dir.glob("*.json"))
        vla_dir = self.root / "configs" / "vla_tasks"
        if vla_dir.is_dir():
            task_paths.extend(sorted(vla_dir.glob("*.json")))
        for path in task_paths:
            definition = load_json(path)
            if not isinstance(definition, dict) or "task_id" not in definition:
                continue
            runtime = definition.get("runtime", definition)
            if definition.get("status") in {"Deprecated", "RelationVariant"} or runtime.get("canonical_task_id") == "place_directional":
                continue
            payload = normalize_task_config(definition, config_root=self.root / "configs")
            spec = self._to_spec(payload)
            if spec.task_id in self._task_specs:
                raise ValueError(f"Duplicate runtime registration_id: {spec.task_id!r}")
            identity = (spec.public_task_id, spec.condition, spec.layout, spec.version, spec.interface)
            for existing in self._task_specs.values():
                if identity == (existing.public_task_id, existing.condition, existing.layout, existing.version, existing.interface) and spec.task_variant == existing.task_variant and spec.parameters == existing.parameters:
                    raise ValueError(f"Duplicate task definition identity: {identity!r}")
            self._task_specs[spec.task_id] = spec
            self._definitions[spec.task_id] = deepcopy(definition)
            self._names.setdefault(spec.public_task_id, []).append(spec.task_id)
            self._payloads[spec.task_id] = dict(payload)
            self._paths[spec.task_id] = path
            aliases = payload.get("aliases", [])
            for alias in aliases:
                alias_key = str(alias).strip()
                if alias_key:
                    self._aliases[alias_key] = spec.task_id
            if spec.canonical_task_id and payload.get("register_canonical_alias", True):
                self._aliases[spec.canonical_task_id] = spec.task_id
            if spec.legacy_task_id and payload.get("register_canonical_alias", True):
                self._aliases[spec.legacy_task_id] = spec.task_id

    def ensure_loaded(self) -> None:
        if not self._task_specs:
            self.load()

    def resolve_task_id(
        self, task_id: str, *, condition: str | None = None, layout: str | None = None,
        version: int | None = None, interface: str | None = None,
        task_variant: str | None = None, parameters: dict[str, Any] | None = None,
    ) -> str:
        self.ensure_loaded()
        key = str(task_id).strip()
        # A renamed bare task alias selects the same conditions/interfaces as
        # its new public name; versioned aliases continue selecting one config.
        if key not in self._names and key in self._aliases and not is_public_task_id(key) and key[:1].isupper():
            key = self._task_specs[self._aliases[key]].public_task_id
        if key not in self._names and any(value is not None for value in (condition, layout, version, interface, task_variant, parameters)):
            resolved = self.resolve_task_id(key)
            spec = self._task_specs[resolved]
            matches = (
                (condition is None or spec.condition == normalize_condition(condition))
                and (layout is None or spec.layout.lower() == layout.lower())
                and (version is None or spec.version == int(version))
                and (interface is None or spec.interface.lower() == interface.lower())
                and (task_variant is None or spec.task_variant.lower() == task_variant.lower())
                and (parameters is None or all(spec.parameters.get(k) == v for k, v in parameters.items()))
            )
            if not matches:
                raise ValueError(f"Task alias {task_id!r} conflicts with the requested selectors.")
            return resolved
        if key in self._names:
            candidates = [self._task_specs[value] for value in self._names[key]]
            if task_variant is not None:
                candidates = [spec for spec in candidates if spec.task_variant.lower() == task_variant.lower()]
            if parameters is not None:
                candidates = [spec for spec in candidates if all(spec.parameters.get(k) == v for k, v in parameters.items())]
            if condition is not None:
                wanted = normalize_condition(condition)
                candidates = [spec for spec in candidates if spec.condition == wanted]
            if layout is not None:
                candidates = [spec for spec in candidates if spec.layout.lower() == layout.lower()]
            if version is not None:
                candidates = [spec for spec in candidates if spec.version == int(version)]
            if interface is not None:
                candidates = [spec for spec in candidates if spec.interface.lower() == interface.lower()]
            elif any(spec.interface == "Standard" for spec in candidates):
                candidates = [spec for spec in candidates if spec.interface == "Standard"]
            if condition is None and any(spec.condition == "Base" for spec in candidates):
                candidates = [spec for spec in candidates if spec.condition == "Base"]
            if version is None and candidates:
                latest = max(spec.version for spec in candidates)
                candidates = [spec for spec in candidates if spec.version == latest]
            if len(candidates) == 1:
                return candidates[0].task_id
            if not candidates:
                raise KeyError(f"No task definition matches {task_id!r} and the requested selectors.")
            raise ValueError(f"Ambiguous task definition {task_id!r}; specify condition/layout/version/interface/task_variant/parameters.")
        if key in self._task_specs:
            return key
        if key in self._aliases:
            return self._aliases[key]
        # Expand legacy family/split shorthand before resolving stored aliases.
        shorthand_candidates: list[str] = []
        if "BaseBase_" not in key and "Base_" in key:
            shorthand_candidates.append(key.replace("Base_", "BaseBase_", 1))
        if "AugBase_" not in key and "Aug_" in key:
            shorthand_candidates.append(key.replace("Aug_", "AugBase_", 1))
        for candidate in shorthand_candidates:
            if candidate in self._task_specs:
                return candidate
            if candidate in self._aliases:
                return self._aliases[candidate]
        raise KeyError(f"Unknown task_id: {task_id!r}")

    def get(self, task_id: str) -> TaskSpec:
        self.ensure_loaded()
        return self._task_specs[self.resolve_task_id(task_id)]

    def get_payload(self, task_id: str) -> dict[str, Any]:
        self.ensure_loaded()
        return dict(self._payloads[self.resolve_task_id(task_id)])

    def get_definition(self, task_id: str, **selectors: Any) -> dict[str, Any]:
        self.ensure_loaded()
        return deepcopy(self._definitions[self.resolve_task_id(task_id, **selectors)])

    def get_compat_definition(self, task_id: str, **selectors: Any) -> dict[str, Any]:
        """Explicit old-runtime view; public definitions remain compact v3."""
        definition = self.get_definition(task_id, **selectors)
        if definition.get("schema_version") == 3:
            from .episode_compat import legacy_definition
            return legacy_definition(definition, self.root / "configs")
        return definition

    def capabilities(self, task_id: str) -> dict[str, Any]:
        from geomind.episodes.resolver import TASKS
        from geomind.episodes.physical_contract import PHYSICS_TASKS
        spec = self.get(task_id)
        memory = spec.task_family in {"SceneUpdating", "IdentityTracking"}
        native=bool(spec.meta.get('native_assembly'))
        return {"task_id": spec.public_task_id,
                "domain": "SpatialMemory" if memory else "SpatialReasoning",
                "semantic": native or spec.public_task_id in TASKS,
                "isaaclab": spec.implementation_status != "placeholder" or spec.public_task_id in PHYSICS_TASKS,
                "isaaclab_legacy": spec.implementation_status != "placeholder" and not native,
                "isaaclab_episode": native or spec.public_task_id in PHYSICS_TASKS,
                "isaaclab_assembled":native}

    def get_config_path(self, task_id: str) -> Path:
        self.ensure_loaded()
        return self._paths[self.resolve_task_id(task_id)]

    def list_task_ids(self, *, include_placeholders: bool = True) -> list[str]:
        self.ensure_loaded()
        return sorted(
            task_id for task_id, spec in self._task_specs.items()
            if include_placeholders or spec.implementation_status != "placeholder"
        )

    def select_by_split(self, split: str) -> list[str]:
        self.ensure_loaded()
        normalized = str(split).strip().lower()
        return sorted(
            spec.task_id for spec in self._task_specs.values()
            if spec.split == normalized and spec.implementation_status != "placeholder"
        )

    def register_gym_envs(self) -> tuple[str, ...]:
        self.ensure_loaded()
        registered: list[str] = []
        for task_id in self.list_task_ids():
            if task_id not in gym.registry:
                gym.register(
                    id=task_id,
                    entry_point="geomind.core.envs.task_env:GeoMindTaskEnv",
                    kwargs={
                        "task_id": task_id,
                        "env_cfg_entry_point": "geomind.core.loader:load_env_cfg_for_task",
                    },
                    disable_env_checker=True,
                )
            registered.append(task_id)
        return tuple(registered)


registry = GeoMindRegistry()
