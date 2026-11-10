# GeoMind Manual Tuning Guide

## Scope

This guide is for manual tuning of existing GeoMind state refresh and geometric structure task configuration.

It assumes the current protocol path stays unchanged:

`annotation.protocol -> task_spec["protocol"] -> cfg.task_protocol`

This guide does not propose:

- a new schema system
- a new engine
- solver changes
- scene/profile chain rewrites
- automatic parameter tuning

The goal is to help you change the right file, at the right layer, with the smallest possible blast radius.

## 1. Configuration Entry Points

### Task-level entry points

- `geomind_lab/mimic/utils/config_semantics_adapter.py`
  Purpose: canonical task registry, raw `task_id` alias mapping, task/scene/instance normalization, lightweight scene metadata overrides.
- `assets/annotations/annotated_tasks.json`
  Purpose: the main hand-editable source of concrete annotated layouts and task entries that are loaded at runtime.
- `geomind_lab/tasks/manager_based/manipulation/geomind/task_specs.py`
  Purpose: loads `annotated_tasks.json`, resolves `layout_key` plus `task_index`, extracts `annotation.protocol`, and remaps object ids onto the runtime scene objects.

### Scene-level entry points

- `geomind_lab/tasks/manager_based/manipulation/geomind/geomind_scene_cfg.py`
  Purpose: physical scene carrier chain. `get_scene_profile(scene_id)` parses the USD scene, and `apply_scene(scene_id)` loads table, basket, chair, lamp, and other assets.
- `geomind_lab/tasks/manager_based/manipulation/geomind/geomind_env_cfg.py`
  Purpose: binds runtime `scene_id` to the environment, then loads task spec and protocol for that scene.
- `geomind_lab/mimic/utils/config_semantics_adapter.py`
  Purpose: `scene_semantics`, capability tags, risk tags, and lightweight scene metadata for selection and analysis.

### Instance-level entry points

- `assets/annotations/annotated_tasks.json`
  Purpose: per-layout `scene_structure`, `object_positions`, task instruction, and per-task annotation payload.
- `scripts/generate_state_reindexing_matrix.py`
  Purpose: state refresh batch instance generation. Defines state refresh variant defaults such as `script_name`, `target_rank`, `order_axis`, `visual_mode`, `intervention_type`, and `intervention_steps`.
- `scripts/generate_geometric_inference_matrix.py`
  Purpose: geometric structure batch instance generation. Defines anchor count, spacing, noise, trend/control settings, success tolerances, and generated query text.

### Benchmark manifest entry points

- `configs/geomind_benchmark_smoke.json`
- `configs/geomind_benchmark_main.json`
- `configs/geomind_benchmark_control.json`
- `configs/geomind_benchmark_stress.json`
  Purpose: suite freeze files. They decide which tasks and agents run together. They can keep old `task_names` and may also include `canonical_task_ids`.
- `geomind_lab/mimic/utils/benchmark_manifest_utils.py`
  Purpose: resolves raw task names and canonical task ids, expands them into concrete task entries, and splits tasks by runner family.
- `scripts/run_geomind_benchmark.sh`
  Purpose: consumes manifest files, resolves canonical task ids into raw tasks, and records selected task descriptors into benchmark info.

### Summary and analysis entry points

- `scripts/summarize_state_reindexing_results.py`
  Purpose: state refresh summary backflow. Enriches results with canonical task id and task/scene/instance semantics.
- `scripts/summarize_geometric_inference_results.py`
  Purpose: geometric structure summary backflow. Enriches results with canonical task id and task/scene/instance semantics.
- `geomind_lab/mimic/utils/joint_summary_utils.py`
  Purpose: common row normalization across state refresh and geometric structure.
- `scripts/summarize_geomind_benchmark_results.py`
  Purpose: joint benchmark summary output and CSV export.

## 2. Recommended Manual Tuning Workflow

### Rule of thumb

Tune in this order:

1. Choose the canonical task.
2. Choose the scene carrier.
3. Tune the concrete instance.
4. Update manifests only if suite membership should change.

If you reverse this order, task, scene, and instance semantics start bleeding into each other again.

### Step 1: Choose the canonical task first

Use `geomind_lab/mimic/utils/config_semantics_adapter.py` as the semantic source of truth.

Pick the canonical task before touching any layout:

- State refresh:
  `state_reindexing_extreme_after_edge_shift`
  `state_reindexing_internal_insertion`
  `state_reindexing_subset_middle_with_distractors`
  `historical_identity_tracking_single_step`
  `historical_identity_tracking_multistep`
- Geometric structure:
  `perfect_midpoint`
  `isometric_extension`
  `trend_continuation`
  `weak_destruction_control`

What to edit at this step:

- `canonical_task_id`
- task-level query intent
- task-level diagnostic tags
- task-level difficulty axes

Where to edit:

- semantic meaning and aliasing: `geomind_lab/mimic/utils/config_semantics_adapter.py`
- one-off concrete task text: `assets/annotations/annotated_tasks.json`
- family-wide generated defaults: generator scripts

What not to edit:

- `layout_key` as if it were the task name
- `scene_id` as if it defined the task
- solver logic

### Step 2: Choose the scene carrier second

Decide whether you are changing:

- scene metadata only
- the runtime physical scene id
- the actual underlying scene asset chain

Recommended interpretation:

- `scene_semantics` is lightweight metadata for human selection and analysis.
- runtime `scene_id` is what actually decides which physical scene gets loaded.
- `geomind_scene_cfg.py` is the physical scene chain and should stay stable unless you truly need a new carrier scene.

Important runtime note:

`geomind_env_cfg.py` loads the runtime scene through `self.scene_id`, then `task_specs.py` loads the task by `layout_key` and `task_index`.
This means annotation-side `scene_id` is a useful default and bookkeeping field, but the runner can still override the actual runtime scene carrier.

What to edit at this step:

- `scene_id`
- `scene_semantics.capability_tags`
- `scene_semantics.risk_tags`
- human-facing scene notes

Where to edit:

- per-layout scene metadata: attach `scene_semantics` in annotated layout records or generated output
- shared scene metadata defaults: `geomind_lab/mimic/utils/config_semantics_adapter.py`
- actual scene asset carrier: `geomind_lab/tasks/manager_based/manipulation/geomind/geomind_scene_cfg.py` only if really necessary

What not to edit:

- task semantics to compensate for a poor scene choice
- canonical task id to encode scene properties

### Step 3: Tune the instance third

This is the layer where `layout`, counts, spacing, noise, intervention, and concrete query phrasing belong.

Use `assets/annotations/annotated_tasks.json` when:

- you are tuning one task or a small number of tasks
- you want a single hand-maintained layout
- you want to preserve existing generator behavior

Use generator scripts when:

- you are tuning a whole family of instances
- you want repeatable structured sweeps
- you want new generated variants with consistent defaults

What to edit at this step:

- `layout`
- `object_count`, `anchor_count`, `distractor_count`
- `visual_mode`
- `spacing`
- `axis_variant`
- `noise_level`, `sigma_parallel`, `sigma_perpendicular`
- `intervention_type`, `intervention_steps`
- `query_template`
- `success_tolerance_l2`, `success_tolerance_orthogonal`
- `difficulty_factors`

Where to edit:

- one-off layouts: `assets/annotations/annotated_tasks.json`
- State refresh family defaults: `scripts/generate_state_reindexing_matrix.py`
- Geometric structure family defaults: `scripts/generate_geometric_inference_matrix.py`

What not to edit:

- canonical task id to encode layout
- scene metadata to encode distractor count or spacing
- protocol engine code when a layout-level field is enough

### Step 4: Update benchmark manifests last

Only edit benchmark manifests if the suite definition itself should change.

Examples:

- add a new raw task to `main`
- swap a control task in `control`
- expand `stress` to include a new canonical task family

Use manifests for selection, not for authoring task semantics.

Files:

- `configs/geomind_benchmark_smoke.json`
- `configs/geomind_benchmark_main.json`
- `configs/geomind_benchmark_control.json`
- `configs/geomind_benchmark_stress.json`

Safe edits:

- `task_names`
- `canonical_task_ids`
- `agent_names`
- `default_agent_groups`
- `summary_preset`
- suite-level `diagnostic_tags`

Unsafe edits for manual tuning:

- trying to encode scene geometry or protocol changes directly inside manifest files

### Practical editing decision table

| If you want to change... | Preferred file |
| --- | --- |
| One task instruction or one task query | `assets/annotations/annotated_tasks.json` |
| Canonical alias mapping or task semantic labels | `geomind_lab/mimic/utils/config_semantics_adapter.py` |
| One-off layout positions or object count | `assets/annotations/annotated_tasks.json` |
| Family-wide state refresh intervention defaults | `scripts/generate_state_reindexing_matrix.py` |
| Family-wide geometric structure spacing/noise defaults | `scripts/generate_geometric_inference_matrix.py` |
| Scene metadata tags for selection and analysis | `geomind_lab/mimic/utils/config_semantics_adapter.py` or layout `scene_semantics` |
| Actual physical scene asset chain | `geomind_lab/tasks/manager_based/manipulation/geomind/geomind_scene_cfg.py` |
| Which tasks belong to smoke/main/control/stress | benchmark manifest JSON files |

## 3. Field Reference

### Task layer fields

| Field | What it does | Preferred edit layer | Most critical for |
| --- | --- | --- | --- |
| `task_id` / `canonical_task_id` | `task_id` is the raw historical label. `canonical_task_id` is the normalized semantic prototype. | Task | All state refresh and geometric structure. Do not encode layout or scene into the canonical id. |
| `family` | High-level benchmark family such as `state_reindexing`, `historical_identity_tracking`, `explicit_rule_extrapolation`, or `implicit_trend_fitting`. | Task | All tasks. Used by generators, manifests, and summaries. |
| `task_type` | Broad execution type such as `discrete_selection` or `continuous_placement`. | Task | state refresh vs geometric structure split, summary grouping, downstream analysis. |
| `core_skill` | Human-readable capability label. | Task | Most useful for analysis and curation across mixed suites. |
| `default_query_template` | Canonical query text for the task prototype. | Task | Especially useful when building new instances or cleaning naming drift. |
| `diagnostic_tags` | Tags describing what the task is testing. | Task | All tasks. Valuable for suite curation and summary slicing. |
| `difficulty_axes` | Which knobs meaningfully change difficulty. | Task | Most useful for deciding which instance fields to tune. |
| `canonical_role` | Usually `task_prototype`; keeps canonical semantics distinct from concrete instances. | Task | Mostly bookkeeping, but useful when inspecting normalized rows. |

### Scene layer fields

| Field | What it does | Preferred edit layer | Most critical for |
| --- | --- | --- | --- |
| `scene_id` | Runtime scene carrier selector. This is what the environment ultimately loads. | Scene | All tasks. Physical carrier choice. |
| `scene_name` | Human-facing scene label. | Scene | Analysis and manual review. |
| `capability_tags` | What kinds of tasks the scene is good at carrying. | Scene | Task-scene matching and benchmark planning. |
| `risk_tags` | Known hazards such as reflection, occlusion, edge risk, or ambiguous camera geometry. | Scene | Stress/control design and failure analysis. |
| `table_shape` | Human-facing note about the support surface geometry. | Scene | More useful for geometric structure tasks and edge-sensitive state refresh tasks. |
| `table_material` | Human-facing note about surface material. | Scene | Useful when reflection or texture affects perception. |
| `background_level` | Coarse complexity level of background clutter. | Scene | More relevant to distractor-heavy state refresh or perception-sensitive geometric structure tasks. |
| `reflection_level` | Coarse reflection risk level. | Scene | Especially relevant to geometry tasks and identical-object state refresh variants. |
| `texture_strength` | Strength of directional or strong visual texture. | Scene | Trend tasks and subset filtering tasks. |
| `workspace_margin` | How much clean tabletop margin is available. | Scene | Insertion, extension, distractor filtering, and shell-game tasks. |
| `camera_profile` | Human-facing note about camera view character. | Scene | Analysis, failure clustering, and task-scene fit. |

Important note:

`scene_semantics` is currently metadata. It does not replace `scene_id` and does not load assets by itself.

### Instance layer fields

| Field | What it does | Preferred edit layer | Most critical for |
| --- | --- | --- | --- |
| `layout` | Concrete spatial pattern such as `linear`, `grid`, `diagonal_line`, or distractor-heavy layouts. | Instance | All tasks. Keep this here, not in canonical task names. |
| `object_count` / `anchor_count` / `distractor_count` | Concrete object cardinalities for the instance. | Instance | state refresh uses `object_count` and `distractor_count`. geometric structure uses `anchor_count`. |
| `visual_mode` | Whether objects are visually distinct or identical. | Instance | state refresh reindexing and identity tracking. |
| `spacing` | Distance between anchors or objects. | Instance | geometric structure midpoint, extension, and trend tasks. |
| `axis_variant` | Axis or row/column flavor such as `x`, `y`, `vertical`, `top_row`, `left_column`. | Instance | state refresh order-axis tasks and geometric structure extension variants. |
| `noise_level` | Coarse noise preset. | Instance | geometric structure trend tasks and weak destruction control. |
| `sigma_parallel` / `sigma_perpendicular` | Fine-grained trend perturbation magnitudes. | Instance | geometric structure trend tasks. |
| `intervention_type` / `intervention_steps` | What scripted change happens and how many steps it takes. | Instance | state refresh reindexing and historical tracking. |
| `query_template` | Concrete per-instance query text. | Instance | Any hand-authored layout where wording matters. |
| `success_tolerance_l2` / `success_tolerance_orthogonal` | Placement success thresholds. | Instance | geometric structure placement tasks. |
| `difficulty_factors` | Captured snapshot of the concrete instance knobs. | Instance | All tasks. Useful for summaries and later filtering. |

## 4. Canonical Task Tuning Notes

The recommendations below combine current canonical scene capability tags with current generator behavior and existing task phrasing.

### `state_reindexing_extreme_after_edge_shift`

- Recommended scene capability tags: `linear_friendly`
- Recommended scene traits: clean tabletop margin, stable left-right ordering, low occlusion, clear extreme positions
- Recommended instance layouts: `linear`; 3 to 5 targets; optional distinct or identical objects
- Avoid: grid-like ambiguity, tight workspace near table edges, heavy distractor clutter unless that is the explicit point of the task
- Representative query: `Pick up the current leftmost object.`
- Best three fields to tune first:
  - `intervention_type`
  - `intervention_steps`
  - `visual_mode`

### `state_reindexing_internal_insertion`

- Recommended scene capability tags: `linear_friendly`, `insertion_friendly`
- Recommended scene traits: enough tabletop margin to insert an object into the middle without accidental collisions, visually stable ordering axis
- Recommended instance layouts: `linear`; 3 to 5 targets; clear internal slots
- Avoid: narrow tables, cluttered backgrounds, layouts where insertion changes are visually ambiguous
- Representative query: `Pick up the object that is currently second from the start.`
- Best three fields to tune first:
  - `object_count`
  - `intervention_type`
  - `axis_variant`

### `state_reindexing_subset_middle_with_distractors`

- Recommended scene capability tags: `linear_friendly`, `subset_filtering_friendly`
- Recommended scene traits: visible separation between target subset and distractors, enough margin to prevent accidental overlap, readable target cues
- Recommended instance layouts: `linear`, `front_targets_back_distractors`, or other layouts that preserve a clear target subset
- Avoid: identical-object scenes without a robust target cue, cramped workspace, distractors interleaved so tightly that subset membership becomes a perception-only problem
- Representative query: `Among the marked target objects, pick up the current leftmost one.`
- Best three fields to tune first:
  - `distractor_count`
  - `visual_mode`
  - `layout`

### `historical_identity_tracking_single_step`

- Recommended scene capability tags: `identity_tracking_friendly`, `linear_friendly`
- Recommended scene traits: clean ordering axis, low occlusion, enough space for one clear move, easy before/after comparison
- Recommended instance layouts: `linear`; 3 targets; one scripted move
- Avoid: scenes with ambiguous camera perspective, distractor-heavy layouts, multi-step shell-game behavior
- Representative query: `Pick the object that was originally in the middle.`
- Best three fields to tune first:
  - `intervention_steps`
  - `axis_variant`
  - `visual_mode`

### `historical_identity_tracking_multistep`

- Recommended scene capability tags: `identity_tracking_friendly`
- Recommended scene traits: clear temporal state changes, enough margin for multiple relocations, minimal occlusion during intermediate states
- Recommended instance layouts: `linear`; 5 targets; multi-step relocation
- Avoid: scenes where intermediate moves run close to basket or furniture boundaries, or where final-frame shortcuts become too easy
- Representative query: `Pick the object that was originally leftmost.`
- Best three fields to tune first:
  - `intervention_steps`
  - `object_count`
  - `layout`

### `perfect_midpoint`

- Recommended scene capability tags: `geometry_friendly`
- Recommended scene traits: stable camera, low reflection, simple background, visually clean anchor pair
- Recommended instance layouts: runtime-injected `blocks`; 2 colored anchors plus 1 target block; zero or minimal noise
- Avoid: reusing scene-native graspable objects, strong perspective ambiguity, busy textures near anchors, distractor-heavy scenes that do not help midpoint reasoning
- Representative query: `Put the grasped object exactly at the midpoint between anchor A and anchor B.`
- Current repo instances that are the most practical for direct debugging:
  - baseline: `scene_id=65`, `layout_key=geometric_inference_perfect_midpoint_line_2`
  - carry-over variant: `scene_id=10`, `layout_key=geometric_inference_midpoint_grid_2_2_top_row`
- Recommended debug entry points:
  - `bash scripts/debug_perfect_midpoint.sh`
  - `PRESET=grid_top_row HEADLESS=1 bash scripts/debug_perfect_midpoint.sh`
  - `bash scripts/geometric_structure_runners/run_geometric_inference_perfect_midpoint_primary.sh`
- Best three fields to tune first:
  - `spacing`
  - `scene_id`
  - `success_tolerance_l2`

### `isometric_extension`

- Recommended scene capability tags: `geometry_friendly`, `extension_friendly`
- Recommended scene traits: clean line continuation, enough outer margin beyond anchor B, low edge risk
- Recommended instance layouts: `linear`, `vertical`, `top_row`, `left_column`
- Avoid: scenes where the extension lands near table boundaries, or layouts where row/column orientation is visually unclear
- Representative query: `Put the object on the extension of the line from A to B with the same spacing.`
- Best three fields to tune first:
  - `spacing`
  - `axis_variant`
  - `scene_id`

### `trend_continuation`

- Recommended scene capability tags: `geometry_friendly`, `trend_friendly`
- Recommended scene traits: enough anchor count to expose a trend, low reflection, stable camera, moderate free space ahead of the trend
- Recommended instance layouts: `linear` or gently noisy line layouts with 3 to 5 anchors
- Avoid: grid-like layouts, overly aggressive perpendicular noise, very tight workspace
- Representative query: `Continue the geometric trend by placing the grasped object at the next position.`
- Best three fields to tune first:
  - `anchor_count`
  - `noise_level`
  - `sigma_perpendicular`

### `weak_destruction_control`

- Recommended scene capability tags: `geometry_friendly`, `trend_friendly`
- Recommended scene traits: same basic carrier quality as `trend_continuation`, but robust enough that weak structural damage can still be analyzed cleanly
- Recommended instance layouts: noisy line layouts with explicit perturbation and enough anchors to show degradation
- Avoid: scenes that are already ambiguous before the weak-destruction perturbation is applied
- Representative query: `Continue the latent trend despite weak structural corruption.`
- Best three fields to tune first:
  - `noise_level`
  - `sigma_perpendicular`
  - `anchor_count`

## 5. Recommended Manual Template

### Recommended format

Use:

- one Markdown guide for human process
- one JSON scratch template for copy-paste editing

Why JSON is the best fit here:

- the current repo is already JSON-heavy for annotations and manifests
- the template can look structurally similar to the fields already emitted by generators and summaries
- it stays easy to diff in version control
- it does not introduce a new runtime dependency

Template file:

- `configs/manual_task_scene_instance_template.json`

Important:

- this template is manual-only
- nothing in the runtime loads it
- it is a planning and review aid, not a new source-of-truth system

### Suggested usage

1. Copy the template into a scratch file next to the task you want to tune.
2. Fill in the canonical task, scene metadata, and instance fields.
3. Apply the chosen values back into the actual runtime-owned files:
   - annotation file
   - generator script
   - manifest file
4. Run the existing generators or summaries as needed.

## 6. Manual Tuning Checklist

Use this checklist after every hand edit.

- `canonical_task_id` still resolves through `config_semantics_adapter.py`
- raw `task_id` aliases were not deleted unless you intentionally migrated all references
- `layout` is still treated as instance-level metadata, not task identity
- scene capability tags still match the intended task family
- scene risk tags still describe the scene honestly
- the chosen `scene_id` still makes sense for the target canonical task
- if a runner or manifest overrides `scene_id`, you accounted for that explicitly
- `annotation.protocol` is still present and structurally valid
- task query text still matches the canonical task semantics
- state refresh intervention fields still match the intended scripted behavior
- geometric structure spacing and noise fields still match the intended difficulty
- generator output remains parseable
- summary scripts still expose `canonical_task_id`, `task_semantics`, `scene_semantics`, and `instance_semantics`
- benchmark manifests were updated only if suite membership changed
- changes to `main`, `control`, or `stress` only changed task selection, not protocol logic

## 7. Changes Not Recommended

Do not do the following unless you are intentionally starting a larger migration project.

- Do not put `layout` back into the canonical task name.
- Do not make one scene the semantic owner of a task family.
- Do not edit solver logic when a task/scene/instance field already exists.
- Do not rewrite the protocol path.
- Do not treat manifest files as the place to author task semantics.
- Do not collapse task, scene, and instance fields back into one naming scheme.
- Do not force a full migration of old annotations just to use canonical semantics.
- Do not delete old raw `task_id` values that existing logs, manifests, or scripts still reference.
- Do not edit `geomind_scene_cfg.py` just to change scene metadata tags.
- Do not use `scene_semantics` as if it were the physical scene loader.
- Do not use `canonical_task_id` to encode scene id, layout family, or instance difficulty.

## 8. Safe Starting Points

If you need the smallest safe place to start, use these:

- One-off task or layout tuning:
  `assets/annotations/annotated_tasks.json`
- Family-wide state refresh tuning:
  `scripts/generate_state_reindexing_matrix.py`
- Family-wide geometric structure tuning:
  `scripts/generate_geometric_inference_matrix.py`
- Canonical semantic cleanup:
  `geomind_lab/mimic/utils/config_semantics_adapter.py`
- Suite composition only:
  benchmark manifest JSON files

That split keeps manual tuning maintainable without reopening the old task/scene/layout entanglement.
