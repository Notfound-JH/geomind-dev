#!/usr/bin/env python3
"""
Single-model OpenPI remote policy smoke test — WebSocket only (aligned with openpi/examples/libero/main.py).

Config JSON (any of):
  - remote_policy_url as ws(s)://…
  - openpi_websocket_url as ws(s)://…
  - host + port (libero Args style) → ws://host:port
  - remote_policy_url as http(s)://… with model_name openpi* → ws://host:port inferred (no HTTP POST)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import numpy as np


def _client_safe_url(url: str, kind: str) -> str:
    """0.0.0.0 / :: are bind-all; clients must use 127.0.0.1 or a concrete IP."""
    p = urlparse(url)
    h = (p.hostname or "").strip().lower()
    if h not in ("0.0.0.0", "::"):
        return url
    nh = "127.0.0.1"
    auth = ""
    if p.username:
        auth = p.username + (f":{p.password}" if p.password else "") + "@"
    netloc = f"{auth}{nh}:{p.port}" if p.port is not None else f"{auth}{nh}"
    out = urlunparse((p.scheme, netloc, p.path or "", p.params, p.query, p.fragment))
    if out != url:
        print(
            f"[openpi-client] note: rewritten {kind} host from {h!r} to {nh!r} "
            "(0.0.0.0 is not a valid client destination; use 127.0.0.1 or the GPU LAN IP)"
        )
    return out


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def build_libero_style_element(*, resize_size: int, prompt: str) -> dict:
    from openpi_client import image_tools

    dummy_rgb = np.zeros((resize_size, resize_size, 3), dtype=np.uint8)
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(dummy_rgb, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(dummy_rgb, resize_size, resize_size))
    state = np.zeros((8,), dtype=np.float32)
    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": str(prompt),
    }


def run_openpi_websocket_like_libero(
    *,
    uri_or_host: str,
    ws_port: int | None,
    element: dict,
    replan_steps: int,
) -> None:
    from openpi_client import websocket_client_policy as _websocket_client_policy

    if uri_or_host.startswith("ws"):
        client = _websocket_client_policy.WebsocketClientPolicy(uri_or_host, None)
    else:
        if ws_port is None:
            raise SystemExit(f"error: need port for websocket host={uri_or_host!r}")
        print(f"[openpi-client] ws_target=ws://{uri_or_host}:{ws_port}")
        client = _websocket_client_policy.WebsocketClientPolicy(uri_or_host, ws_port)

    meta = client.get_server_metadata()
    print(
        "[openpi-client] server_metadata_keys="
        f"{sorted(meta.keys()) if isinstance(meta, dict) else type(meta).__name__}"
    )

    use_broker = _env_bool("OPENPI_USE_ACTION_CHUNK_BROKER", False)
    print(f"[openpi-client] libero_style_single_infer={not use_broker} replan_steps={replan_steps}")

    if use_broker:
        from openpi_client.action_chunk_broker import ActionChunkBroker

        broker = ActionChunkBroker(client, action_horizon=replan_steps)
        for step in range(replan_steps):
            out = broker.infer(element)
            keys = sorted(out.keys()) if isinstance(out, dict) else []
            print(f"[openpi-client] ws step={step} action_keys={keys}")
        return

    out = client.infer(element)
    if not isinstance(out, dict) or "actions" not in out:
        keys = sorted(out.keys()) if isinstance(out, dict) else None
        raise SystemExit(
            "error: infer() must return dict with 'actions' (openpi/examples/libero/main.py); "
            f"type={type(out).__name__} keys={keys}"
        )
    action_chunk = np.asarray(out["actions"], dtype=np.float32)
    if action_chunk.ndim == 1:
        action_chunk = action_chunk.reshape(1, -1)
    nchunk = int(action_chunk.shape[0])
    if nchunk < replan_steps:
        raise SystemExit(
            f"error: len(actions) first dim is {nchunk} but need >= replan_steps={replan_steps} "
            "(libero asserts same vs replan_steps)"
        )
    for step in range(replan_steps):
        row = action_chunk[step]
        print(f"[openpi-client] ws step={step} action_shape={row.shape} dtype={row.dtype}")


def _resolve_openpi_websocket_url(*, cfg: dict, config_path: str) -> tuple[str, bool]:
    """Returns (ws_url, inferred_from_http_remote)."""
    remote_policy_url = str(cfg.get("remote_policy_url", "")).strip()
    ws_url_cfg = str(cfg.get("openpi_websocket_url", "")).strip()
    ws_url_env = str(os.environ.get("OPENPI_WS_URL", "")).strip()
    ws_url = ws_url_cfg or ws_url_env
    inferred = False
    mn = str(cfg.get("model_name", "")).lower()

    host_key = str(cfg.get("host", "")).strip()
    port_raw = cfg.get("port", None)
    if not ws_url and not remote_policy_url and host_key and port_raw is not None and str(port_raw).strip() != "":
        try:
            pnum = int(port_raw)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"error: invalid port in config host/port: {port_raw!r}") from exc
        ws_url = f"ws://{host_key}:{pnum}"
        inferred = True
        print(f"[openpi-client] note: synthesized WebSocket from host+port (libero-style): {ws_url!r}")

    if remote_policy_url:
        remote_policy_url = _client_safe_url(remote_policy_url, "remote_policy_url")
    if ws_url:
        ws_url = _client_safe_url(ws_url, "openpi_websocket_url")

    scheme = ""
    if remote_policy_url:
        scheme = (urlparse(remote_policy_url).scheme or "").lower()

    if not ws_url and remote_policy_url and scheme in ("ws", "wss"):
        ws_url = remote_policy_url
        inferred = True
        print(f"[openpi-client] note: using remote_policy_url as WebSocket: {ws_url!r}")

    if not ws_url and remote_policy_url and scheme in ("http", "https") and mn.startswith("openpi"):
        p = urlparse(remote_policy_url)
        host = p.hostname or "127.0.0.1"
        port = p.port
        if port is None:
            raise SystemExit(f"error: need port in remote_policy_url to infer WebSocket: {remote_policy_url!r}")
        ws_url = f"ws://{host}:{port}"
        inferred = True
        print(f"[openpi-client] note: inferred WebSocket {ws_url!r} from http-shaped remote_policy_url")

    if not ws_url:
        raise SystemExit(
            f"error: could not resolve WebSocket URL from {config_path} "
            "(set remote_policy_url ws(s)://, openpi_websocket_url, host+port, or http(s):// for openpi*)"
        )

    return ws_url, inferred


def dispatch_one_config(config_path: str, chunk_len: int, prompt: str) -> None:
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model_name = str(cfg.get("model_name", "unknown"))
    ws_url, ws_inferred = _resolve_openpi_websocket_url(cfg=cfg, config_path=config_path)

    resize = int(os.environ.get("OPENPI_RESIZE_SIZE", "224"))
    element = build_libero_style_element(resize_size=resize, prompt=prompt)

    print(f"[openpi-client] model_name={model_name}")
    print(f"[openpi-client] transport=websocket_only")
    print(f"[openpi-client] openpi_websocket_effective={ws_url!r} inferred={ws_inferred}")
    print(f"[openpi-client] action_chunk_exec_len={chunk_len}")

    run_openpi_websocket_like_libero(uri_or_host=ws_url, ws_port=None, element=element, replan_steps=chunk_len)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) < 3:
        print(
            "usage: openpi_client_all_models_main.py CONFIG.json CHUNK_LEN PROMPT",
            file=sys.stderr,
        )
        return 2
    config_path, chunk_len_raw, prompt = argv[0], argv[1], argv[2]
    chunk_len = int(chunk_len_raw)
    dispatch_one_config(config_path, chunk_len, prompt)
    print("[openpi-client] communication_test=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
