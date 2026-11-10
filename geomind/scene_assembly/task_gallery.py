"""Publish assembled-task evidence to the existing local atlas, without uploads.

This is a researcher-facing view. Ground truth is copied only into its explicit
privileged evidence panel, never into the task's public observation artifact.
"""
from geomind_lab.utils.portable_paths import scene_atlas_root
from pathlib import Path
import hashlib
import json
import shlex
import shutil

from .assets import ROOT


def publish_task_local(output):
    """Copy actual artifacts and atomically update the local task index."""
    output = Path(output).resolve()
    site = scene_atlas_root() / "dist"
    if not site.is_dir():
        return None
    result = json.loads((output / "result.json").read_text())
    task = result.get("task")
    if not isinstance(task, dict):
        task = json.loads((output / "task_result.json").read_text())
    manifest_path = Path(result["assembly_manifest"]).resolve()
    manifest = json.loads(manifest_path.read_text())
    request = manifest["request"]
    identity = output.name + "-" + hashlib.sha256(str(output).encode()).hexdigest()[:8]
    target = site / "assets/assembled_tasks" / identity
    target.mkdir(parents=True, exist_ok=True)
    prefix = "assets/assembled_tasks/" + identity + "/"
    urls = {}
    for key in ("preview", "detail_preview", "video"):
        value = manifest.get(key) or manifest.get("physics", {}).get(key)
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = manifest_path.parent / path
        if path.is_file():
            filename = key + path.suffix
            shutil.copyfile(path, target / filename)
            urls[key] = prefix + filename
    shutil.copyfile(output / "result.json", target / "result.json")
    shutil.copyfile(manifest_path, target / "assembly_manifest.json")
    for key in ("public_observation_path", "ground_truth_path"):
        if task.get(key):
            path = Path(task[key])
            if not path.is_absolute():
                path = output / path
            if path.is_file():
                filename = key + ".json"
                shutil.copyfile(path, target / filename)
                urls[key.replace("_path", "_url")] = prefix + filename
    request_path = next((path for path in (output / "request.json", output / "raw_request.json") if path.is_file()), None)
    command = None
    if request_path:
        replay_request = json.loads(request_path.read_text())
        replay_request['output'] = str(output.with_name(output.name + '_replay'))
        replay_path = output / 'replay_request.json'
        replay_path.write_text(json.dumps(replay_request, ensure_ascii=False, indent=2))
        shutil.copyfile(replay_path, target / 'replay_request.json')
        urls['replay_request_url'] = prefix + 'replay_request.json'
        command = shlex.join(["python", "-m", "geomind.tools.research_pipeline", "--assembled-request", str(replay_path)])
    record = {
        "id": identity,
        "status": result.get("status"),
        "selection": result.get("selection", {}),
        "scene_id": request["scene_id"],
        "scene_name": manifest.get("scene", {}).get("scene_name"),
        "assembly_layout": request["layout"],
        "asset": request["object"]["asset"],
        "count": request["object"]["count"],
        "seed": request.get("seed"),
        "randomization": request.get("randomization", {}),
        "task_id": task.get("task_id"),
        "operation": task.get("operation"),
        "instruction": task.get("instruction"),
        "ground_truth": task.get("ground_truth"),
        "query_source": task.get("source", {}),
        "checks": task.get("checks", {}),
        "objects": task.get("objects", []),
        "statuses": {
            "gt_generated": result.get("gt_verified"),
            "initial_layout_physics_passed": result.get("initial_layout_physics_verified"),
            "task_action_physics_executed": result.get("task_action_physics_executed"),
            "task_action_physics_passed": result.get("task_action_physics_verified"),
            "model_evaluated": result.get("model_evaluated"),
        },
        "physics": manifest.get("physics", {}),
        "result_url": prefix + "result.json",
        "manifest_url": prefix + "assembly_manifest.json",
        "manifest_path": str(manifest_path),
        "result_path": str(output / "result.json"),
        "native_config_root": task.get("native_config_root"),
        "episode_id": task.get("episode_id"),
        "command": command,
        **urls,
    }
    robot = result.get('robot_execution')
    if isinstance(robot,dict):
        report=robot.get('report',{})
        robot_rows=[]
        for case in report.get('results',[]):
            name=case['episode_id']+('_negative' if case.get('negative_control') else '')
            media={}
            for key,field in [('observer','video_url'),('last_frame','preview_url')]:
                path=Path((case.get('video') or {}).get(key,''))
                if path.is_file():
                    filename='robot_'+name+'_'+key+path.suffix
                    shutil.copyfile(path,target/filename);media[field]=prefix+filename
            case_path=Path(robot['report_path']).parent/'cases'/f'{name}.json'
            if case_path.exists():
                filename='robot_'+name+'.json';shutil.copyfile(case_path,target/filename);media['report_url']=prefix+filename
            robot_rows.append({key:case.get(key) for key in ['negative_control','success','verification_passed','instruction','ground_truth','physical']}|media)
        record['robot_execution']={'robot':robot.get('robot'),'action':robot.get('action'),
             'executed':robot.get('executed'),'passed':robot.get('passed'),
             'positive_successes':report.get('positive_successes'),'positive_cases':report.get('positive_cases'),
             'negative_controls':report.get('negative_controls'),'offline_audit':robot.get('offline_audit'),
             'cases':robot_rows}
        positive=next((row for row in report.get('results',[]) if not row.get('negative_control')),None)
        if positive:
            record['runtime_ground_truth']=positive['ground_truth']
            record['runtime_instruction']=positive['instruction']
            record['runtime_query_clock']=positive.get('query_physical_clock')
    index = site / "assets/assembled_tasks.json"
    rows = json.loads(index.read_text()).get("tasks", []) if index.exists() else []
    rows = [row for row in rows if row.get("id") != identity]
    rows.insert(0, record)
    rows.sort(key=lambda row:(row.get('statuses',{}).get('task_action_physics_passed') is not True,
                              row.get('statuses',{}).get('task_action_physics_executed') is not True))
    temporary = index.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"schema_version": 1, "tasks": rows}, ensure_ascii=False, indent=2))
    temporary.replace(index)
    return "http://127.0.0.1:8780/tasks.html"
