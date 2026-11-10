"""Explicit agent/user choices for binding assembled scenes to native task GT.

This is a strict request boundary and an inventory menu, not an LLM or a second
task registry. The actor supplies its selection and reason; the binding layer
loads the actual assembly, checks object identities and solves native contracts.
"""
from __future__ import annotations
from geomind_lab.utils.portable_paths import repo_assets_root

from collections.abc import Mapping
import json
import math

from geomind.episodes.resolver import TASKS

from .assets import ROOT, resolve_asset, resolve_scene
from .request import ASSET_IDS, LAYOUT_TYPES, RequestError, validate_request


_SUPPORTED_TASK_IDS = tuple(name for name in TASKS if name in {
    "SelectByOrder", "InferMidpoint", "ProjectOntoLine", "ReflectAcrossLine",
})


def _object(value, label, *, required, optional=()):
    if not isinstance(value, Mapping):
        raise RequestError(f"{label} must be an object")
    missing=set(required)-set(value)
    extra=set(value)-set(required)-set(optional)
    if missing:
        raise RequestError(f"Missing {label} fields: {sorted(missing)}")
    if extra:
        raise RequestError(f"Unknown {label} fields: {sorted(map(str,extra))}")
    return value


def _text(value, label):
    if not isinstance(value,str) or not value.strip() or "\0" in value:
        raise RequestError(f"{label} must be a nonempty string without null bytes")
    return value.strip()


def _identifier(value, label):
    value=_text(value,label)
    if not all(c.isascii() and (c.isalnum() or c in "_-") for c in value):
        raise RequestError(f"{label} must be an ASCII object ID")
    return value


def _positive(value, label):
    try:
        valid=type(value) in (int,float) and math.isfinite(value) and value>0
    except OverflowError:
        valid=False
    if not valid:
        raise RequestError(f"{label} must be a finite positive number")
    return float(value)


def _vector3(value, label):
    try:
        valid=isinstance(value,list) and len(value)==3 and all(type(item) in (int,float) and math.isfinite(item) for item in value)
    except OverflowError:
        valid=False
    if not valid:
        raise RequestError(f"{label} must contain three finite numbers")
    if any(abs(item) > 10 for item in value):
        raise RequestError(f"{label} must use world metres within [-10,10]")
    return [float(item) for item in value]


def validate_robot_execution(robot_raw, execution_raw, task, *, timeout_s=420):
    """Validate explicit assembled manipulation, with no reach-only fallback."""
    robot_raw=_object(robot_raw,"robot",required={"embodiment","base_position_w_m"},optional={"active_arm"})
    name=robot_raw["embodiment"]
    robots={"franka","fr3","franka_robotiq","piper","piper_dual"}
    if not isinstance(name,str) or name not in robots:
        raise RequestError(f"Unknown robot embodiment: {name!r}")
    robot={"embodiment":name,"base_position_w_m":_vector3(robot_raw["base_position_w_m"],"robot.base_position_w_m")}
    if "active_arm" in robot_raw:
        if name!="piper_dual" or robot_raw["active_arm"] not in ("left","right"):
            raise RequestError("active_arm is left/right and only applies to piper_dual")
        robot["active_arm"]=robot_raw["active_arm"]
    elif name=="piper_dual":
        raise RequestError("piper_dual requires explicit active_arm")
    fields={"negative_controls","timeout_s","target_offset_m","position_tolerance_m","orientation_tolerance_rad",
            "dwell_steps","max_steps","joint_step_limit_rad","approach_clearance_m"}
    execution_raw=_object(execution_raw,"execution",required={"action"},optional=fields)
    expected="PickLiftSelectedObject" if task.get('operation',{}).get('type')=='pick' or task["task_id"]=="SelectByOrder" else "PickPlaceObject"
    if execution_raw["action"]!=expected:
        raise RequestError(f"Natural operation requires execution.action={expected}; ReachAboveTarget is diagnostic only")
    if name not in {"franka","franka_robotiq"}:
        raise RequestError(f"Assembled pick/place execution for {name} is not implemented or verified; historical target reaching is not a grasp fallback")
    negative=execution_raw.get("negative_controls",True)
    if type(negative) is not bool:
        raise RequestError("execution.negative_controls must be a boolean")
    timeout=_positive(execution_raw.get("timeout_s",timeout_s),"execution.timeout_s")
    if not 1<=timeout<=420:
        raise RequestError("execution.timeout_s must be in 1..420 seconds per worker")
    execution={"action":expected,"negative_controls":negative,"timeout_s":timeout,
        "target_offset_m":[0.,0.,.1],"position_tolerance_m":.01,"orientation_tolerance_rad":.15,
        "dwell_steps":60,"max_steps":12000 if name=="franka_robotiq" else 3000,
        "joint_step_limit_rad":.04,"approach_clearance_m":.15}
    ranges={"position_tolerance_m":(.001,.03),"orientation_tolerance_rad":(.01,.5),
            "joint_step_limit_rad":(.001,.1),"approach_clearance_m":(.06,.3)}
    for field,(low,high) in ranges.items():
        value=_positive(execution_raw.get(field,execution[field]),f"execution.{field}")
        if not low<=value<=high:
            raise RequestError(f"execution.{field} must be in {low}..{high}")
        execution[field]=value
    for field,low,high in (("dwell_steps",60,600),("max_steps",120,12000)):
        value=execution_raw.get(field,execution[field])
        if type(value) is not int or not low<=value<=high:
            raise RequestError(f"execution.{field} must be an integer in {low}..{high}")
        execution[field]=value
    if execution["max_steps"]<execution["dwell_steps"]:
        raise RequestError("execution.max_steps must cover dwell_steps")
    if "target_offset_m" in execution_raw:
        offset=_vector3(execution_raw["target_offset_m"],"execution.target_offset_m")
        if offset[:2]!=[0.,0.] or not .01<=offset[2]<=.3:
            raise RequestError("execution.target_offset_m must be [0,0,z] with z in .01..0.3 metres; unused by pick/place")
        execution["target_offset_m"]=offset
    return robot,execution


def _identifiers(value, label, *, count=None):
    if not isinstance(value,list) or not value:
        raise RequestError(f"{label} must be a nonempty list of object IDs")
    result=[_identifier(item,label) for item in value]
    if len(set(result))!=len(result):
        raise RequestError(f"{label} must not repeat an object ID")
    if count is not None and len(result)!=count:
        raise RequestError(f"{label} must contain exactly {count} distinct object IDs")
    return result


def _parameters(task_id, raw):
    fields={
        "SelectByOrder":({"index","from","candidates"},{"frame","tie_tolerance_m"}),
        "InferMidpoint":({"endpoints"},{"frame"}),
        "ProjectOntoLine":({"point","line"},{"frame","geometry","degeneracy_tolerance_m"}),
        "ReflectAcrossLine":({"point","line"},{"frame","geometry","degeneracy_tolerance_m"}),
    }
    required,optional=fields[task_id]
    raw=_object(raw,"task.parameters",required=required,optional=optional)
    if raw.get("frame","world")!="world":
        raise RequestError("Assembled tasks require frame='world'; no implicit frame transform")
    result={"frame":"world"}
    if task_id=="SelectByOrder":
        candidates=_identifiers(raw["candidates"],"task.parameters.candidates")
        index=raw["index"]
        if type(index) is not int or not 1<=index<=len(candidates):
            raise RequestError("task.parameters.index must be an integer in 1..len(candidates)")
        if raw["from"] not in ("Left","Right"):
            raise RequestError("task.parameters.from must be Left or Right")
        result.update(index=index,**{"from":raw["from"]},candidates=candidates,
                      tie_tolerance_m=_positive(raw.get("tie_tolerance_m",.01),"tie_tolerance_m"))
    elif task_id=="InferMidpoint":
        result["endpoints"]=_identifiers(raw["endpoints"],"task.parameters.endpoints",count=2)
    else:
        result["point"]=_identifier(raw["point"],"task.parameters.point")
        result["line"]=_identifiers(raw["line"],"task.parameters.line",count=2)
        if result["point"] in result["line"]:
            raise RequestError("Point object must differ from line endpoints")
        if raw.get("geometry","InfiniteLine3D")!="InfiniteLine3D":
            raise RequestError("Only native InfiniteLine3D geometry is supported")
        result["geometry"]="InfiniteLine3D"
        result["degeneracy_tolerance_m"]=_positive(raw.get("degeneracy_tolerance_m",.0001),"degeneracy_tolerance_m")
    return result


def validate_task_request(raw):
    """Return a fresh canonical request; never choose a hidden task or layout.

    Object-ID existence and settled geometry are checked by the binding layer.
    Task operation intent is explicit, so semantic GT cannot be labelled a
    completed pick/place operation by this request parser.
    """
    raw=_object(raw,"assembled task request",required={"output","task","selection"},
                optional={"schema_version","stage","assembly","assembly_manifest","timeout_s","robot","execution"})
    version=raw.get("schema_version",1)
    if type(version) is not int or version!=1:
        raise RequestError("assembled task schema_version must be 1")
    stage=raw.get("stage","assemble_and_solve_gt")
    if not isinstance(stage,str) or stage not in {"assemble_and_solve_gt","assemble_solve_execute"}:
        raise RequestError("Unknown assembled-task stage; choose assemble_and_solve_gt or assemble_solve_execute")
    if ("assembly" in raw)==("assembly_manifest" in raw):
        raise RequestError("Provide exactly one of assembly or assembly_manifest")
    timeout=_positive(raw.get("timeout_s",420),"timeout_s")
    if timeout<1 or timeout>420:
        raise RequestError("timeout_s must be in 1..420 seconds per worker")
    result={"schema_version":1,"stage":stage,"output":_text(raw["output"],"output"),"timeout_s":timeout}
    if "assembly" in raw:
        result["assembly"]=validate_request(raw["assembly"])
        try:
            resolve_asset(result["assembly"]["object"]["asset"])
            resolve_scene(result["assembly"]["scene_id"])
        except ValueError as exc:
            raise RequestError(str(exc)) from exc
    else:
        result["assembly_manifest"]=_text(raw["assembly_manifest"],"assembly_manifest")
    task=_object(raw["task"],"task",required={"task_id","parameters","operation"})
    task_id=task["task_id"]
    if not isinstance(task_id,str) or task_id not in _SUPPORTED_TASK_IDS:
        raise RequestError(f"Static assembled task unsupported: {task_id!r}; supported: {_SUPPORTED_TASK_IDS}")
    parameters=_parameters(task_id,task["parameters"])
    operation=_object(task["operation"],"task.operation",required={"type"},optional={"moving_object"})
    operation_type=operation["type"]
    expected="pick" if task_id=="SelectByOrder" else "place"
    if operation_type!=expected:
        raise RequestError(f"{task_id} requires explicit operation.type={expected!r}")
    normalized_operation={"type":operation_type}
    if "moving_object" in operation:
        moving=_identifier(operation["moving_object"],"task.operation.moving_object")
        references=parameters.get("endpoints",parameters.get("line",[]))+([parameters["point"]] if "point" in parameters else [])
        if moving in references:
            raise RequestError("moving_object cannot be a reference object used to solve the target")
        normalized_operation["moving_object"]=moving
    elif expected=="place":
        raise RequestError("place requires task.operation.moving_object")
    result["task"]={"task_id":task_id,"parameters":parameters,"operation":normalized_operation}
    selection=_object(raw["selection"],"selection",required={"actor","reason","considered_scene_ids","considered_layouts"})
    if selection["actor"] not in ("agent","user"):
        raise RequestError("selection.actor must be agent or user")
    scenes=selection["considered_scene_ids"]
    if not isinstance(scenes,list) or not scenes or any(type(s) is not int or not 1<=s<=102 for s in scenes) or len(scenes)!=len(set(scenes)):
        raise RequestError("selection.considered_scene_ids must be distinct scene IDs in 1..102")
    layouts=selection["considered_layouts"]
    if not isinstance(layouts,list) or not layouts or any(not isinstance(kind,str) or kind not in LAYOUT_TYPES for kind in layouts) or len(layouts)!=len(set(layouts)):
        raise RequestError(f"selection.considered_layouts must be distinct supported layouts: {LAYOUT_TYPES}")
    if "assembly" in result:
        assembly=result["assembly"]
        if assembly["scene_id"] not in scenes or assembly["layout"]["type"] not in layouts:
            raise RequestError("The selected scene and layout must occur in the recorded considered options")
    result["selection"]={"actor":selection["actor"],"reason":_text(selection["reason"],"selection.reason"),
                         "considered_scene_ids":list(scenes),"considered_layouts":list(layouts)}
    if stage=="assemble_solve_execute":
        if "robot" not in raw or "execution" not in raw:
            raise RequestError("assemble_solve_execute requires explicit robot and execution")
        result["robot"],result["execution"]=validate_robot_execution(raw["robot"],raw["execution"],result["task"],timeout_s=timeout)
    elif "robot" in raw or "execution" in raw:
        raise RequestError("robot/execution require stage='assemble_solve_execute'; GT-only cannot silently execute actions")
    return result


def assembly_options():
    """Return available local scene/asset inventory and existing static GT API."""
    index=json.loads((repo_assets_root()/"scenes/tabletop_cleared_v1/scene_index.json").read_text())
    scenes=[]
    for row in index["scenes"]:
        resolved=resolve_scene(row["scene_id"])
        scenes.append({**row,"usd_path":resolved["usd_path"]})
    assets=[];excluded=[]
    for name in ASSET_IDS:
        try:
            assets.append(resolve_asset(name))
        except ValueError as exc:
            excluded.append({"name":name,"reason":str(exc)})
    hints={
        "SelectByOrder":{"required_parameters":["index","from","candidates"],
            "optional_parameters":{"frame":"world","tie_tolerance_m":.01},"operation":"pick",
            "compatible_layouts":["line"],"layout_hint":"Use line axis=x; grid rows can be tied on X. Geometry still requires a unique order."},
        "InferMidpoint":{"required_parameters":["endpoints"],"optional_parameters":{"frame":"world"},
            "operation":"place","compatible_layouts":list(LAYOUT_TYPES),
            "layout_hint":"Choose two distinct references and a separate moving object; target must pass support/occupancy checks."},
        "ProjectOntoLine":{"required_parameters":["point","line"],
            "optional_parameters":{"frame":"world","geometry":"InfiniteLine3D","degeneracy_tolerance_m":.0001},
            "operation":"place","compatible_layouts":["grid","random","t_shape","ring"],
            "layout_hint":"Use a nondegenerate reference line, an off-line point and a separate moving object."},
        "ReflectAcrossLine":{"required_parameters":["point","line"],
            "optional_parameters":{"frame":"world","geometry":"InfiniteLine3D","degeneracy_tolerance_m":.0001},
            "operation":"place","compatible_layouts":["grid","random","t_shape","ring"],
            "layout_hint":"Native semantics are axial reflection/180-degree rotation around an infinite 3D line, not unrestricted 2D mirror semantics."},
    }
    tasks=[{"task_id":name,"domain":TASKS[name][0],"family":TASKS[name][1],"response":TASKS[name][2],
            "solver":"geomind.episodes.resolver.solve",**hints[name]} for name in _SUPPORTED_TASK_IDS]
    # Keep the four static request schemas above unchanged. Registered native
    # manipulation has its own explicit existing batch entry point; adding it
    # to this inventory must not silently accept new fields in the static API.
    from geomind.core.registry import GeoMindRegistry
    from geomind.episodes.registered_tasks import supported_registered_task_ids, REGISTERED_TASK_IDS
    registrations=GeoMindRegistry();registrations.load()
    registered_tasks=[]
    for identity in supported_registered_task_ids():
        spec=registrations.get(identity)
        action=(spec.meta.get('native_assembly',{}).get('action')
                or ('PickPlaceObject' if spec.task_family=='GeometricInference'
                    or spec.public_task_id=='PlaceByTopology' else 'PickLiftSelectedObject'))
        registered_tasks.append({'registration_id':identity,'task_id':spec.public_task_id,
            'domain':spec.domain,'family':spec.task_family,'action':action,
            'registration_kind':'historical_registered_adapter' if identity in REGISTERED_TASK_IDS else 'AssembledNative',
            'implementation_status':spec.implementation_status,
            'runner_entrypoint':'geomind.tools.regenerate_registered_tasks',
            'command_template':f'python -m geomind.tools.regenerate_registered_tasks --tasks {identity} --output NEW_DIRECTORY --max-steps 12000',
            'robot':'franka_robotiq','worker_timeout_s':420,
            'evidence_requirement':'Actual initial settling, runtime GT, true pick/place, requested negative controls and independent raw-state audit must pass per instance; registration is not execution evidence'})
    layouts=[{"type":kind,"constraints":{
        "grid":"Explicit rows*cols must equal count; row axis=X, column axis=Y.",
        "line":"axis is x or y; default y, SelectByOrder requires x for a robust full order.",
        "ring":"Conservative circular footprint and bounded radius.",
        "random":"Seeded bounded rejection sampling; never silently reduces count.",
        "t_shape":"Exactly five objects.",
    }[kind]} for kind in LAYOUT_TYPES]
    return {"schema_version":1,"selection_actor":"agent or user supplies the decision; this menu does not invoke an LLM",
            "scene_count":len(scenes),"scenes":scenes,"asset_count":len(assets),"assets":assets,"excluded_assets":excluded,
            "layout_count":len(layouts),"layouts":layouts,"task_count":len(tasks),"tasks":tasks,
            "registered_manipulation_task_count":len(registered_tasks),
            "registered_manipulation_canonical_type_count":len({task['task_id'] for task in registered_tasks}),
            "registered_manipulation_tasks":registered_tasks,
            "object_id_pattern":"<canonical_asset>_001 ... <canonical_asset>_<count padded to 3 digits>",
            "public_object_ids":"Request references use source asset IDs; the evaluated agent sees independently permuted semantic IDs and presentation order. The private evaluator stores the binding.",
            "state_contract":"Solve from the actual assembly query snapshot in world metres, with no second layout jitter.",
            "robot_options":[{"embodiment":name,"historical_actions":["ReachAboveTarget"],
                "assembled_manipulation_actions":["PickLiftSelectedObject","PickPlaceObject"] if name in {"franka","franka_robotiq"} else [],
                "assembly_validation_status":"implementation_available; every generated instance needs physical validation" if name in {"franka","franka_robotiq"} else "assembled grasp not implemented or verified"}
                for name in ("franka","fr3","franka_robotiq","piper","piper_dual")],
            "stages":["assemble_and_solve_gt","assemble_solve_execute"],
            "physical_status":"Initial assembly settling, task semantic GT and robot pick/place execution are verified separately."}
