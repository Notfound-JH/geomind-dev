"""Decode, audit, freeze and package a recorded native physical video dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from geomind.episodes.resolver import IMPLEMENTATION_HASH, _check, _fingerprint, load_json
from geomind.tools.migrate_episode_configs import write


def inspect_video(path: Path, rows: list[dict], *, kind: str, config: dict) -> dict:
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(path))
    _check(cap.isOpened(), f"Cannot decode video: {path}")
    count, masked, first, last = 0, 0, None, None
    try:
        _check(abs(cap.get(cv2.CAP_PROP_FPS) - config["fps"]) < 1e-5, f"Video fps mismatch: {path}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _check(count < len(rows), f"Extra video frame: {path}")
            _check(frame.shape == (config["height"], config["width"], 3), f"Resolution mismatch: {path}")
            hidden = kind == "agent" and not rows[count]["scene_visible_to_agent"]
            if hidden:
                _check(int(frame.max()) <= 3, f"Hidden memory scene leaked into agent video: {path}, frame {count}")
                masked += 1
            else:
                _check(float(frame.std()) > 1, f"Unexpected blank frame: {path}, frame {count}")
            if first is None:
                first = frame.copy()
            last = frame
            count += 1
    finally:
        cap.release()
    _check(count == len(rows) and count > 1, f"Truncated video: {path}")
    difference = float(np.abs(last.astype(float) - first.astype(float)).mean())
    _check(difference > 0.1, f"Video does not show scene/robot motion: {path}")
    return {"decoded_frames": count, "masked_frames": masked, "first_last_mean_pixel_difference": difference}


def package(root: Path, configs: Path, archive: Path) -> dict:
    import numpy as np
    root = root.resolve()
    report = load_json(root / "report.json")
    _check(report["physical_execution_verified"] and report["video_cases"] == len(report["results"]),
           "Dataset requires a completed physical run with a video for every case")
    entries, decoded, masked = [], 0, 0
    for case in report["results"]:
        _check(case["provenance"]["source_hashes"]["implementation"] == IMPLEMENTATION_HASH,
               "Video run used another implementation; freeze its original sources instead")
        name = case["episode_id"] + ("_negative" if case["negative_control"] else "")
        video = case["video"]
        rows = [json.loads(line) for line in Path(video["frames"]).read_text().splitlines()]
        _check(len(rows) == video["frame_count"], f"Frame metadata count mismatch: {name}")
        dt = case["query_physical_clock"]["dt_s"]
        stride = round(1 / (video["fps"] * dt))
        for index, row in enumerate(rows):
            _check(row["frame_index"] == index, f"Nonconsecutive frame indices: {name}")
            _check(row["physics_tick"] == video["start_physics_tick"] + index * stride, f"Timestamp gap: {name}")
            _check(abs(row["video_time_s"] - row["episode_time_s"]) < 1e-10, f"Video speed/timestamp mismatch: {name}")
        _check(rows[-1]["physics_tick"] <= case["physics_steps"] < rows[-1]["physics_tick"] + stride,
               f"Video did not cover the complete execution: {name}")
        if case["domain"] == "SpatialMemory":
            observe_end = next(iter(case["query_physical_clock"]["history_ticks"].values()))
            query_tick = case["query_physical_clock"]["current_tick"]
            _check(all(r["scene_visible_to_agent"] == (not observe_end <= r["physics_tick"] < query_tick) for r in rows),
                   f"Agent visibility does not match protocol boundaries: {name}")
        checks = {}
        for kind in ("observer", "agent"):
            path = Path(video[kind])
            _check(hashlib.sha256(path.read_bytes()).hexdigest() == video["sha256"][kind], f"Video checksum mismatch: {path}")
            checks[kind] = inspect_video(path, rows, kind=kind, config=video)
            decoded += checks[kind]["decoded_frames"]
        masked += checks["agent"]["masked_frames"]
        _check(hashlib.sha256(Path(video["frames"]).read_bytes()).hexdigest() == video["sha256"]["frames"],
               f"Frame metadata checksum mismatch: {name}")
        with np.load(case["trace"]) as data:
            trace, columns = data["data"], data["columns"].tolist()
            _check(np.isfinite(trace).all(), f"Nonfinite physical trace: {name}")
            for row in rows:
                sample = trace[row["physics_tick"] - 1]
                _check(int(sample[0]) == row["physics_tick"], f"Video/trajectory alignment mismatch: {name}")
        episode = load_json(configs / "episodes" / f"{case['episode_id']}.json")
        refs = {"episode": Path("episodes") / f"{episode['episode_id']}.json",
                "task": Path("task_definitions") / f"{episode['task_id']}.json",
                "scene": Path("scenes") / f"{episode['scene_id']}.json",
                "layout": Path("layouts") / f"{episode['layout_id']}.json"}
        if "protocol_id" in episode:
            refs["protocol"] = Path("protocols") / f"{episode['protocol_id']}.json"
        for key, relative in refs.items():
            _check(_fingerprint(load_json(configs / relative)) == case["provenance"]["source_hashes"][key],
                   f"Source config changed since recording: {relative}")
            target = root / "configs" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(configs / relative, target)
        layout = load_json(configs / refs["layout"])
        labels = {o["id"]: o.get("label", o["id"]) for o in layout["objects"]}
        encoding = None
        if case["domain"] == "SpatialMemory":
            encoding = "Remember the locations of " + ", ".join(labels[o] for o in episode["parameters"]["objects"]) + "."
        query_tick = case["query_physical_clock"]["current_tick"]
        entries.append({"episode_id": name, "task_id": case["task_id"], "domain": case["domain"],
            "seed": episode["seed"], "success": case["success"], "negative_control": case["negative_control"],
            "observer_video": str(Path(video["observer"]).resolve().relative_to(root)),
            "agent_video": str(Path(video["agent"]).resolve().relative_to(root)),
            "frame_metadata": str(Path(video["frames"]).resolve().relative_to(root)),
            "trajectory": str(Path(case["trace"]).resolve().relative_to(root)),
            "case_report": f"cases/{name}.json", "evidence": f"evidence/{name}.json",
            "query_instruction": case["instruction"], "encoding_instruction": encoding,
            "query_start_frame": next(r["frame_index"] for r in rows if r["physics_tick"] >= query_tick),
            "frame_count": len(rows), "duration_s": video["duration_s"], "sha256": video["sha256"], "checks": checks})
    source_root = Path(__file__).resolve().parents[1] / "episodes"
    for path in source_root.glob("*.py"):
        target = root / "source" / "episodes" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    manifest = {"dataset": root.name, "backend": report["backend"], "policy": report["policy"],
        "physical_execution_verified": True, "episode_count": len(entries), "video_count": 2 * len(entries),
        "recording": report["recording"], "total_decoded_frames": decoded, "agent_masked_frames": masked,
        "observer_video_privileged": True, "paths_relative_to_dataset_root": True, "episodes": entries}
    write(root / "dataset_manifest.json", manifest)
    (root / "README.md").write_text(
        "# 无头物理视频数据\n\n"
        "以 dataset_manifest.json 为索引，文件路径均相对于本目录。\n\n"
        "- *_observer.mp4：完整过程视频，包含 memory 隐藏阶段，属于评估/调试数据。\n"
        "- *_agent.mp4：同样的 RGB 帧，memory 隐藏阶段整帧遮黑。模型输入使用这一版本。\n"
        "- *_frames.jsonl：逐帧索引、物理 tick、视频/仿真时间、phase 和可见性。\n"
        "- traces/：逐物理步关节与 TCP/物体状态；用 physics_tick-1 索引对应行。\n"
        "- cases/ 与 evidence/：GT 和评估证据，不可作为 agent 输入。\n"
        "- configs/ 与 source/：本次录制所用的配置和实现快照。\n\n"
        f"视频从 reset 落地稳定结束开始，{report['recording']['fps']} FPS、"
        f"{report['recording']['width']}×{report['recording']['height']}，按实际仿真时间播放，无加速。\n"
        "manifest 中 query_start_frame 为查询指令开始生效的帧；memory 在此前使用 encoding_instruction。\n"
        "所有视频均完成逐帧解码、分辨率/帧数、时间对齐、遮罩和 SHA-256 检查。\n",
        encoding="utf-8")
    archive = archive.resolve()
    _check(not archive.is_relative_to(root), "Archive must be outside the dataset directory")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                bundle.write(path, Path(root.name) / path.relative_to(root))
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(archive.suffix + ".sha256").write_text(f"{checksum}  {archive.name}\n")
    return {"episodes": len(entries), "videos": 2 * len(entries), "decoded_frames": decoded,
            "agent_masked_frames": masked, "archive": str(archive), "archive_bytes": archive.stat().st_size,
            "sha256": checksum}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--configs", type=Path, default=Path(__file__).resolve().parents[1] / "configs")
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    archive = args.archive or args.root.with_suffix(".zip")
    print(json.dumps(package(args.root, args.configs, archive), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
