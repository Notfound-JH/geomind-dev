from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from .policy_runtime import run_episode, save_episode_payload


def run_task(
    *,
    task_id: str,
    max_steps: int,
    output_dir: Path,
    backend: str = "mock",
    interactive: bool = False,
    condition: str | None = None,
    layout: str | None = None,
    version: int | None = None,
    interface: str | None = None,
    task_variant: str | None = None,
    parameters: dict | None = None,
) -> dict[str, Any]:
    from geomind.core.registry import registry
    spec=registry.get(task_id)
    if spec.meta.get('native_assembly'):
        if backend not in {'isaaclab','assembled_native'}:
            raise ValueError('Canonical manipulation requires --backend assembled_native; mock execution is unavailable')
        if interactive or any(value is not None for value in (condition,layout,version,interface,task_variant,parameters)):
            from geomind.core.resolver import resolve_task_request
            resolved=resolve_task_request(task_id,backend=backend,condition=condition,layout=layout,
                version=version,interface=interface,task_variant=task_variant,parameters=parameters)
            task_id=resolved.resolved_task_id
            if interactive:raise ValueError('Assembled-native pipeline does not support interactive manager actions')
        if not 120<=max_steps<=12000:
            raise ValueError('Native manipulation needs --max-steps in 120..12000; use 12000 for this validated contract')
        from geomind.tools.regenerate_registered_tasks import run_batch
        return run_batch(output_dir,tasks=[task_id],scene_id=65,seed=0,max_steps=max_steps,
                         robot_base=[-.12,.32,.9328504589976385])
    result = run_episode(
        task_id=task_id,
        policy_name="oracle",
        max_steps=max_steps,
        backend=backend,
        interactive=interactive,
        condition=condition, layout=layout, version=version, interface=interface,
        task_variant=task_variant, parameters=parameters,
    )
    for key in ("policy_mode", "action_noise_scale", "object_goal_perturb_xy"):
        result.summary.pop(key, None)
    save_episode_payload(output_dir=output_dir, result=result)
    return dict(result.summary)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a single GeoMind task by task_id (no explicit policy field).")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--condition", default=None, help="Task condition, e.g. Base or Visual")
    parser.add_argument("--layout", default=None, help="Layout category, e.g. Line or Cross")
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--interface", choices=("Standard", "Vla"), default=None)
    parser.add_argument("--task-variant", default=None, help="Public task variant, e.g. Inside or Clutter")
    parser.add_argument("--parameters", type=json.loads, default=None, help="Task parameters as a JSON object")
    parser.add_argument("--order-index", type=int, default=None, help="One-based order index")
    parser.add_argument("--order-from", choices=("Left", "Right"), default=None)
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="When enabled, accept policy_action from observation meta if provided.",
    )
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--backend", default="mock")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    parameters = args.parameters
    if parameters is not None and not isinstance(parameters, dict):
        parser.error("--parameters must be a JSON object")
    if args.order_index is not None or args.order_from is not None:
        parameters = dict(parameters or {})
        if args.order_index is not None: parameters["index"] = args.order_index
        if args.order_from is not None: parameters["from"] = args.order_from

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("geomind") / "outputs" / "single" / f"{timestamp}_{args.task_id}"
    result = run_task(
        task_id=args.task_id,
        max_steps=int(args.max_steps),
        output_dir=output_dir,
        backend=str(args.backend),
        interactive=bool(args.interactive),
        condition=args.condition, layout=args.layout, version=args.version, interface=args.interface,
        task_variant=args.task_variant, parameters=parameters,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"result_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
