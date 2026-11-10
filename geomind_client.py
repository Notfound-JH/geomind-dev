from __future__ import annotations

import json
import math
import os
import base64
import time
from io import BytesIO
from typing import Any
from urllib import error, request
from urllib.parse import urlparse

import numpy as np

try:
    from geomind.core.observation import to_agent_ready_observation as _to_agent_ready_observation
except Exception:
    _to_agent_ready_observation = None


DEFAULT_REMOTE_POLICY_URL = "http://127.0.0.1:8000/policy"
DEFAULT_REMOTE_ACTION_CHUNK_LEN = 1
DEFAULT_ENV_STEP_HZ = 24.0
DEFAULT_POLICY_HZ = 10.0
DEFAULT_MAX_XY_NORM_M = 0.0
DEFAULT_MAX_XY_COMPONENT_M = 0.0
DEFAULT_WORKSPACE_BOUNDS = (0.15, 0.75, -0.45, 0.45, 0.70, 1.20)
# IsaacLab 的 DifferentialInverseKinematicsActionCfg(scale=0.5, use_relative_mode=True)
# 会在 env.step() 内部再乘一次，所以下发动作的真实米级位移 = action * ENV_ACTION_SCALE。
DEFAULT_ENV_ACTION_SCALE = 0.5
_DUMMY_IMAGE_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)
_ACTION_CHUNK_CACHE: list[np.ndarray] = []
_EPISODE_HTTP_REQUEST_COUNT = 0
_EPISODE_CACHE_HIT_COUNT = 0
_EPISODE_TOTAL_EFFECTIVE_CHUNK_LEN = 0
_EPISODE_TOTAL_SERVER_CHUNK_LEN = 0
_EPISODE_LAST_ENV_STEP_HZ = float(DEFAULT_ENV_STEP_HZ)
_EPISODE_LAST_PHYSICS_STEP_HZ = 0.0
_EPISODE_LAST_ACTION_REPEAT = 1
_EPISODE_LAST_CONFIGURED_CHUNK_LEN = int(DEFAULT_REMOTE_ACTION_CHUNK_LEN)
_EPISODE_ACTIVE = False
_EPISODE_LAST_ACTION: np.ndarray | None = None
_EPISODE_LAST_GOAL_ERROR_XY: np.ndarray | None = None
_EPISODE_INITIAL_GRIPPER_ACTION: float | None = None
_EPISODE_INITIAL_GRIPPER_POS: list[float] = []
_EPISODE_ID = ""
_LAST_POLICY_DEBUG: dict[str, Any] = {}
_LAST_EFFECTIVE_CHUNK_LEN = 0
_LAST_SERVER_CHUNK_LEN = 0
_OPENPI_WS_CLIENT: Any = None
_OPENPI_WS_CLIENT_URI: str = ""
_OPENPI_STATE_FALLBACK_LOGGED = False


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _remote_policy_action_scale() -> float:
    return float(_env_float("REMOTE_POLICY_ACTION_SCALE", 1.0))


def _configured_remote_action_chunk_len() -> int:
    return max(_env_int("REMOTE_ACTION_CHUNK_LEN", DEFAULT_REMOTE_ACTION_CHUNK_LEN), 1)


def _configured_policy_hz() -> float:
    return max(_env_float("REMOTE_POLICY_HZ", DEFAULT_POLICY_HZ), 0.0)


def _infer_env_step_hz(env: Any, default_hz: float = DEFAULT_ENV_STEP_HZ) -> float:
    candidates = [
        env,
        getattr(env, "unwrapped", None),
        getattr(env, "_runtime_env", None),
        getattr(env, "_native_env", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        step_dt = getattr(candidate, "step_dt", None)
        try:
            step_dt_value = float(step_dt)
            if step_dt_value > 0.0:
                return 1.0 / step_dt_value
        except Exception:
            pass

        cfg = getattr(candidate, "cfg", None)
        sim = getattr(cfg, "sim", None)
        dt = getattr(sim, "dt", None)
        decimation = getattr(cfg, "decimation", None)
        try:
            dt_value = float(dt)
            decimation_value = float(decimation)
            if dt_value > 0.0 and decimation_value > 0.0:
                return 1.0 / (dt_value * decimation_value)
        except Exception:
            continue
    return float(default_hz)


def _iter_env_candidates(env: Any) -> list[Any]:
    candidates = [
        env,
        getattr(env, "unwrapped", None),
        getattr(env, "_runtime_env", None),
        getattr(env, "_native_env", None),
    ]
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None:
        candidates.extend(
            [
                getattr(unwrapped, "_runtime_env", None),
                getattr(unwrapped, "_native_env", None),
            ]
        )
    runtime_env = getattr(unwrapped, "_runtime_env", None) if unwrapped is not None else None
    if runtime_env is not None:
        candidates.append(getattr(runtime_env, "_native_env", None))
    return [candidate for candidate in candidates if candidate is not None]


def _infer_physics_step_hz(env: Any) -> float:
    for candidate in _iter_env_candidates(env):
        physics_dt = getattr(candidate, "physics_dt", None)
        try:
            physics_dt_value = float(physics_dt)
            if physics_dt_value > 0.0:
                return 1.0 / physics_dt_value
        except Exception:
            pass
        cfg = getattr(candidate, "cfg", None)
        sim = getattr(cfg, "sim", None)
        dt = getattr(sim, "dt", None)
        try:
            dt_value = float(dt)
            if dt_value > 0.0:
                return 1.0 / dt_value
        except Exception:
            pass
    return 0.0


def _infer_action_repeat(env: Any) -> int:
    for candidate in _iter_env_candidates(env):
        cfg = getattr(candidate, "cfg", None)
        decimation = getattr(cfg, "decimation", None)
        try:
            value = int(decimation)
            if value > 0:
                return value
        except Exception:
            pass
    physics_hz = _infer_physics_step_hz(env)
    env_hz = _infer_env_step_hz(env, default_hz=DEFAULT_ENV_STEP_HZ)
    if physics_hz > 0.0 and env_hz > 0.0:
        return max(int(round(physics_hz / env_hz)), 1)
    return 1


def _reset_episode_stats(env: Any) -> None:
    global _EPISODE_HTTP_REQUEST_COUNT
    global _EPISODE_CACHE_HIT_COUNT
    global _EPISODE_TOTAL_EFFECTIVE_CHUNK_LEN
    global _EPISODE_TOTAL_SERVER_CHUNK_LEN
    global _EPISODE_LAST_ENV_STEP_HZ
    global _EPISODE_LAST_PHYSICS_STEP_HZ
    global _EPISODE_LAST_ACTION_REPEAT
    global _EPISODE_LAST_CONFIGURED_CHUNK_LEN
    global _EPISODE_ACTIVE
    global _EPISODE_LAST_ACTION
    global _EPISODE_LAST_GOAL_ERROR_XY
    global _EPISODE_INITIAL_GRIPPER_ACTION
    global _EPISODE_INITIAL_GRIPPER_POS
    global _EPISODE_ID
    global _LAST_POLICY_DEBUG
    global _LAST_EFFECTIVE_CHUNK_LEN
    global _LAST_SERVER_CHUNK_LEN

    _ACTION_CHUNK_CACHE.clear()
    _EPISODE_HTTP_REQUEST_COUNT = 0
    _EPISODE_CACHE_HIT_COUNT = 0
    _EPISODE_TOTAL_EFFECTIVE_CHUNK_LEN = 0
    _EPISODE_TOTAL_SERVER_CHUNK_LEN = 0
    _EPISODE_LAST_ENV_STEP_HZ = _infer_env_step_hz(env, default_hz=DEFAULT_ENV_STEP_HZ)
    _EPISODE_LAST_PHYSICS_STEP_HZ = _infer_physics_step_hz(env)
    _EPISODE_LAST_ACTION_REPEAT = _infer_action_repeat(env)
    _EPISODE_LAST_CONFIGURED_CHUNK_LEN = _configured_remote_action_chunk_len()
    _EPISODE_ACTIVE = True
    _EPISODE_LAST_ACTION = None
    _EPISODE_LAST_GOAL_ERROR_XY = None
    _EPISODE_INITIAL_GRIPPER_ACTION = None
    _EPISODE_INITIAL_GRIPPER_POS = []
    _EPISODE_ID = f"ep_{int(time.time() * 1000)}"
    _LAST_POLICY_DEBUG = {}
    _LAST_EFFECTIVE_CHUNK_LEN = 0
    _LAST_SERVER_CHUNK_LEN = 0


def get_remote_policy_debug_snapshot() -> dict[str, Any]:
    return dict(_LAST_POLICY_DEBUG)


def _shape_of(value: Any) -> list[int]:
    try:
        shape = list(np.asarray(value).shape)
    except Exception:
        shape = []
    return [int(dim) for dim in shape]


def _extract_request_id(resp: dict[str, Any], default: str) -> str:
    request_id = resp.get("request_id")
    if isinstance(request_id, str) and request_id.strip():
        return request_id.strip()
    result = resp.get("result")
    if isinstance(result, dict):
        nested = result.get("request_id")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    return default


def get_remote_policy_stats() -> dict[str, Any]:
    avg_chunk_len = (
        float(_EPISODE_TOTAL_EFFECTIVE_CHUNK_LEN) / float(_EPISODE_HTTP_REQUEST_COUNT)
        if _EPISODE_HTTP_REQUEST_COUNT > 0
        else 0.0
    )
    avg_server_chunk_len = (
        float(_EPISODE_TOTAL_SERVER_CHUNK_LEN) / float(_EPISODE_HTTP_REQUEST_COUNT)
        if _EPISODE_HTTP_REQUEST_COUNT > 0
        else 0.0
    )
    estimated_remote_hz = float(_EPISODE_LAST_ENV_STEP_HZ)
    configured_policy_hz = _configured_policy_hz()
    if configured_policy_hz > 0.0 and _EPISODE_LAST_ENV_STEP_HZ > configured_policy_hz:
        query_interval = max(int(round(_EPISODE_LAST_ENV_STEP_HZ / configured_policy_hz)), 1)
        estimated_remote_hz = float(_EPISODE_LAST_ENV_STEP_HZ) / float(query_interval)
    if _env_bool("REMOTE_POLICY_USE_ACTION_CHUNK", default=False):
        estimated_remote_hz = float(_EPISODE_LAST_ENV_STEP_HZ) / float(max(_EPISODE_LAST_CONFIGURED_CHUNK_LEN, 1))
    return {
        "active": bool(_EPISODE_ACTIVE),
        "env_step_hz": float(_EPISODE_LAST_ENV_STEP_HZ),
        "physics_step_hz": float(_EPISODE_LAST_PHYSICS_STEP_HZ),
        "action_repeat": int(_EPISODE_LAST_ACTION_REPEAT),
        "configured_policy_hz": float(configured_policy_hz),
        "remote_action_chunk_len": int(_EPISODE_LAST_CONFIGURED_CHUNK_LEN),
        "estimated_remote_policy_hz": float(estimated_remote_hz),
        "episode_http_request_count": int(_EPISODE_HTTP_REQUEST_COUNT),
        "average_effective_chunk_length": float(avg_chunk_len),
        "average_server_chunk_length": float(avg_server_chunk_len),
        "cache_hit_count": int(_EPISODE_CACHE_HIT_COUNT),
    }


def _safe_to_agent_obs(raw_obs: dict[str, Any]) -> dict[str, Any]:
    if callable(_to_agent_ready_observation):
        try:
            return dict(_to_agent_ready_observation(raw_obs))
        except Exception:
            pass
    timestep = raw_obs.get("timestep", {})
    return {
        "instruction": raw_obs.get("instruction", ""),
        "state": raw_obs.get("state", {}),
        "cameras": raw_obs.get("cameras", {}),
        "episode_step": (timestep or {}).get("episode_step", 0),
    }


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _as_uint8_image(value: Any) -> np.ndarray | None:
    try:
        image = np.asarray(value)
    except Exception:
        return None
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        return None
    if image.dtype != np.uint8:
        image = image.astype(np.float32, copy=False)
        max_value = float(np.nanmax(image)) if image.size else 0.0
        if max_value <= 1.0:
            image = image * 255.0
        image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    return image


def _image_to_b64(value: Any) -> str:
    image = _as_uint8_image(value)
    if image is None:
        return ""
    try:
        from PIL import Image
    except Exception:
        return ""
    with BytesIO() as buffer:
        Image.fromarray(image).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def _find_native_camera(raw_obs: dict[str, Any], camera_name: str) -> Any:
    native_obs = (raw_obs.get("state") or {}).get("native_obs")
    if not isinstance(native_obs, dict):
        return None
    policy_obs = native_obs.get("policy")
    if isinstance(policy_obs, dict) and camera_name in policy_obs:
        return policy_obs.get(camera_name)
    if camera_name in native_obs:
        return native_obs.get(camera_name)
    return None


def _camera_names_from_env() -> list[str]:
    names = [
        name.strip()
        for name in str(os.environ.get("REMOTE_POLICY_CAMERA_NAME", "table_cam,wrist_cam")).split(",")
        if name.strip()
    ]
    return names or ["table_cam", "wrist_cam"]


def _extract_camera_b64(raw_obs: dict[str, Any], camera_name: str) -> str:
    env_key = "REMOTE_POLICY_IMAGE_B64_" + "".join(
        ch.upper() if ch.isalnum() else "_" for ch in str(camera_name)
    )
    explicit_named = str(os.environ.get(env_key, "")).strip()
    if explicit_named:
        return explicit_named

    cameras = raw_obs.get("cameras", {})
    if isinstance(cameras, dict):
        value = cameras.get(camera_name)
        if value is not None:
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, dict):
                for key in ("image_b64", "rgb_b64", "color_b64"):
                    candidate = str(value.get(key, "")).strip()
                    if candidate:
                        return candidate

    return _image_to_b64(_find_native_camera(raw_obs, camera_name))


def _extract_images_b64(raw_obs: dict[str, Any]) -> dict[str, str]:
    images: dict[str, str] = {}
    explicit = str(os.environ.get("REMOTE_POLICY_IMAGE_B64", "")).strip()
    if explicit:
        images[_camera_names_from_env()[0]] = explicit
        return images

    for camera_name in _camera_names_from_env():
        image_b64 = _extract_camera_b64(raw_obs, camera_name)
        if image_b64:
            images[camera_name] = image_b64

    if not images and _env_bool("REMOTE_POLICY_USE_DUMMY_IMAGE"):
        for camera_name in _camera_names_from_env():
            images[camera_name] = _DUMMY_IMAGE_B64
    _dump_images_for_debug(images)
    return images


_IMAGE_DUMP_COUNT = 0


def _dump_images_for_debug(images: dict[str, str]) -> None:
    """Persist the exact images sent to the remote policy when explicitly enabled.

    Controlled by REMOTE_POLICY_IMAGE_DUMP_DIR; REMOTE_POLICY_IMAGE_DUMP_MAX caps
    how many frames are written (default 8) so an enabled dump cannot fill the disk.
    """
    global _IMAGE_DUMP_COUNT
    dump_dir = str(os.environ.get("REMOTE_POLICY_IMAGE_DUMP_DIR", "")).strip()
    if not dump_dir or not images:
        return
    try:
        limit = int(os.environ.get("REMOTE_POLICY_IMAGE_DUMP_MAX", "8"))
    except ValueError:
        limit = 8
    if _IMAGE_DUMP_COUNT >= limit:
        return
    index = _IMAGE_DUMP_COUNT
    _IMAGE_DUMP_COUNT += 1
    try:
        os.makedirs(dump_dir, exist_ok=True)
        for camera_name, image_b64 in images.items():
            safe_name = "".join(ch if ch.isalnum() else "_" for ch in str(camera_name))
            target = os.path.join(dump_dir, f"{index:03d}_{safe_name}.png")
            with open(target, "wb") as handle:
                handle.write(base64.b64decode(image_b64))
    except Exception as exc:  # pragma: no cover - debug-only path
        print(f"[geomind_client] image dump failed: {exc.__class__.__name__}: {exc}")


def _extract_image_b64(raw_obs: dict[str, Any]) -> str:
    explicit = str(os.environ.get("REMOTE_POLICY_IMAGE_B64", "")).strip()
    if explicit:
        return explicit
    images = _extract_images_b64(raw_obs)
    for camera_name in _camera_names_from_env():
        image_b64 = images.get(camera_name, "")
        if image_b64:
            return image_b64
    return ""


def _action_dim_from_env(env: Any, default: int = 7) -> int:
    action_space = getattr(env, "action_space", None)
    shape = getattr(action_space, "shape", None)
    if not shape:
        return int(default)
    try:
        action_dim = int(np.prod(shape))
    except Exception:
        return int(default)
    return max(action_dim, 1)


def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    req = request.Request(
        url=url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=max(float(timeout_s), 0.1)) as resp:
        body = resp.read().decode("utf-8")
    parsed = json.loads(body) if body else {}
    return parsed if isinstance(parsed, dict) else {}


def _remote_policy_url_is_openpi_websocket(url: str) -> bool:
    u = (url or "").strip().lower()
    return u.startswith("ws://") or u.startswith("wss://")


def _openpi_websocket_uri_for_policy() -> str | None:
    """OpenPI uses WebSocket only: for REMOTE_POLICY_MODEL openpi*, never HTTP POST."""
    model = str(os.environ.get("REMOTE_POLICY_MODEL", "")).strip().lower()
    if not model.startswith("openpi"):
        return None
    ru = str(os.environ.get("REMOTE_POLICY_URL", "")).strip()
    ows = str(os.environ.get("OPENPI_WEBSOCKET_URL", "")).strip()
    if ows and _remote_policy_url_is_openpi_websocket(ows):
        return _openpi_ws_connect_uri(ows)
    if _remote_policy_url_is_openpi_websocket(ru):
        return _openpi_ws_connect_uri(ru)
    if ru:
        scheme = (urlparse(ru).scheme or "").lower()
        if scheme in ("http", "https"):
            p = urlparse(ru)
            host = p.hostname
            port = p.port
            if host and port is not None:
                return _openpi_ws_connect_uri(f"ws://{host}:{int(port)}")
    return None


def _openpi_ws_connect_uri(uri: str) -> str:
    """0.0.0.0 is a bind address; clients must use a concrete host."""
    s = str(uri).strip()
    if "0.0.0.0" in s:
        return s.replace("0.0.0.0", "127.0.0.1")
    return s


class _GeomindOpenpiWebsocketClient:
    """Same wire protocol as openpi_client.WebsocketClientPolicy, with bounded wait + stdout progress."""

    def __init__(self, uri: str) -> None:
        import websockets.sync.client

        from openpi_client import msgpack_numpy

        self._uri = _openpi_ws_connect_uri(uri)
        self._packer = msgpack_numpy.Packer()
        api_key = str(os.environ.get("OPENPI_API_KEY", "") or "").strip() or None
        max_wait_s = max(_env_float("OPENPI_WS_CONNECT_MAX_WAIT_S", 120.0), 0.0)
        retry_s = max(_env_float("OPENPI_WS_CONNECT_RETRY_S", 5.0), 0.5)
        open_timeout = max(_env_float("OPENPI_WS_OPEN_TIMEOUT_S", 15.0), 2.0)
        log_connect = _env_bool("OPENPI_WS_CONNECT_LOG", True)
        deadline = time.perf_counter() + max_wait_s
        headers = {"Authorization": f"Api-Key {api_key}"} if api_key else None
        self._ws: Any = None
        self._server_metadata: dict[str, Any] = {}
        attempt = 0
        while True:
            attempt += 1
            if time.perf_counter() >= deadline:
                raise RuntimeError(
                    f"[geomind_client] OpenPI websocket connect to {self._uri!r} timed out after {max_wait_s}s "
                    f"({attempt} attempts). Start the OpenPI policy server and check host/port/firewall. "
                    "Env: OPENPI_WS_CONNECT_MAX_WAIT_S, OPENPI_WS_CONNECT_RETRY_S, OPENPI_WS_OPEN_TIMEOUT_S."
                )
            if log_connect:
                print(
                    f"[geomind_client] openpi_websocket_connect attempt={attempt} uri={self._uri!r}",
                    flush=True,
                )
            try:
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    open_timeout=open_timeout,
                )
                raw = conn.recv()
                if isinstance(raw, str):
                    raise RuntimeError(f"OpenPI websocket server error on handshake:\n{raw}")
                self._server_metadata = msgpack_numpy.unpackb(raw)
                self._ws = conn
                if log_connect:
                    print("[geomind_client] openpi_websocket_connected", flush=True)
                return
            except TimeoutError:
                if log_connect:
                    print(
                        f"[geomind_client] openpi_websocket_handshake_timeout (open_timeout={open_timeout}s); "
                        "retrying…",
                        flush=True,
                    )
            except ConnectionRefusedError:
                if log_connect:
                    print("[geomind_client] openpi_websocket_connection_refused; retrying…", flush=True)
            except OSError as exc:
                errno = getattr(exc, "errno", None)
                if errno in {111, 99, 61}:
                    if log_connect:
                        print(
                            f"[geomind_client] openpi_websocket_os_error errno={errno} {exc!r}; retrying…",
                            flush=True,
                        )
                else:
                    raise
            time.sleep(retry_s)

    def get_server_metadata(self) -> dict[str, Any]:
        return dict(self._server_metadata) if isinstance(self._server_metadata, dict) else {}

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        from openpi_client import msgpack_numpy

        assert self._ws is not None
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def reset(self) -> None:
        pass


def _get_openpi_websocket_client(ws_uri: str) -> Any:
    global _OPENPI_WS_CLIENT, _OPENPI_WS_CLIENT_URI
    ws_uri = _openpi_ws_connect_uri(str(ws_uri).strip())
    if not ws_uri:
        raise ValueError("empty OpenPI websocket uri")
    if _OPENPI_WS_CLIENT is not None and _OPENPI_WS_CLIENT_URI == ws_uri:
        return _OPENPI_WS_CLIENT
    try:
        from openpi_client import websocket_client_policy as _websocket_client_policy
    except ImportError as exc:
        raise RuntimeError(
            "OpenPI remote policy uses WebSocket (ws:// or wss://, or REMOTE_POLICY_MODEL=openpi* with http URL "
            "to derive host:port). Install openpi_client in this Python environment."
        ) from exc
    # Stock client loops forever on refused connections with only logging.info (often invisible here).
    max_wait = _env_float("OPENPI_WS_CONNECT_MAX_WAIT_S", 120.0)
    if max_wait <= 0.0:
        _OPENPI_WS_CLIENT = _websocket_client_policy.WebsocketClientPolicy(ws_uri, None)
    else:
        _OPENPI_WS_CLIENT = _GeomindOpenpiWebsocketClient(ws_uri)
    _OPENPI_WS_CLIENT_URI = ws_uri
    return _OPENPI_WS_CLIENT


def _b64_to_rgb_uint8(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to decode camera images for OpenPI websocket mode") from exc
    with BytesIO(raw) as buf:
        img = Image.open(buf).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _quat2axisangle(quat_xyzw: np.ndarray) -> np.ndarray:
    """robosuite transform_utils.quat2axisangle (xyzw order), as used by openpi/examples/libero/main.py."""
    w = float(np.clip(float(quat_xyzw[3]), -1.0, 1.0))
    den = math.sqrt(max(1.0 - w * w, 0.0))
    if math.isclose(den, 0.0):
        return np.zeros((3,), dtype=np.float32)
    return (np.asarray(quat_xyzw[:3], dtype=np.float32) * 2.0 * math.acos(w) / den).astype(np.float32)


def _build_openpi_state(raw_obs: dict[str, Any], state_dim: int) -> np.ndarray:
    """LIBERO-style proprioception: [eef_pos(3), axisangle(eef_quat)(3), gripper_pos(2)].

    Mirrors openpi/examples/libero/main.py, which concatenates robot0_eef_pos,
    _quat2axisangle(robot0_eef_quat) and robot0_gripper_qpos. IsaacLab reports the end-effector
    quaternion as wxyz while robosuite (and therefore the axis-angle helper) expects xyzw.
    """
    global _OPENPI_STATE_FALLBACK_LOGGED

    native_obs = (raw_obs.get("state") or {}).get("native_obs")
    eef_pos = _as_float_array(_find_key_recursively(native_obs, "eef_pos"), min_len=3)
    eef_quat_wxyz = _as_float_array(_find_key_recursively(native_obs, "eef_quat"), min_len=4)
    gripper_pos = _extract_gripper_pos(raw_obs)

    state = np.zeros((max(state_dim, 8),), dtype=np.float32)
    missing: list[str] = []
    if eef_pos is not None:
        state[0:3] = eef_pos[:3]
    else:
        missing.append("eef_pos")
    if eef_quat_wxyz is not None:
        quat_xyzw = np.asarray(
            [eef_quat_wxyz[1], eef_quat_wxyz[2], eef_quat_wxyz[3], eef_quat_wxyz[0]], dtype=np.float32
        )
        state[3:6] = _quat2axisangle(quat_xyzw)
    else:
        missing.append("eef_quat")
    if gripper_pos is not None:
        width = min(gripper_pos.size, 2)
        state[6 : 6 + width] = gripper_pos[:width]
    else:
        missing.append("gripper_pos")

    if missing and not _OPENPI_STATE_FALLBACK_LOGGED:
        _OPENPI_STATE_FALLBACK_LOGGED = True
        print(
            f"[geomind_client] warning: observation/state missing {missing} in native_obs; "
            "those entries stay zero. OpenPI policies are trained with real proprioception.",
            flush=True,
        )

    return state[:state_dim].astype(np.float32, copy=False)


def _build_openpi_infer_element(env: Any, raw_obs: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Assemble a LIBERO-format OpenPI request.

    openpi/examples/libero/main.py feeds the policy ``obs["agentview_image"][::-1, ::-1]``.
    robosuite renders bottom-up, so the first flip only restores an upright image and the
    second one is a genuine left-right mirror, i.e. LIBERO policies are trained on mirrored
    frames. IsaacLab already renders upright, so matching that convention needs the
    horizontal mirror alone; ``OPENPI_IMAGE_HFLIP``/``OPENPI_IMAGE_VFLIP`` expose both flips
    for A/B checks.
    """
    from openpi_client import image_tools

    resize = max(_env_int("OPENPI_RESIZE_SIZE", 224), 1)
    names = _camera_names_from_env()
    images_b64 = _extract_images_b64(raw_obs)
    hflip = _env_bool("OPENPI_IMAGE_HFLIP", default=True)
    vflip = _env_bool("OPENPI_IMAGE_VFLIP", default=False)

    def cam_rgb(camera_name: str) -> np.ndarray:
        b64 = str(images_b64.get(camera_name, "") or "").strip()
        if not b64:
            blank = np.zeros((resize, resize, 3), dtype=np.uint8)
            return image_tools.convert_to_uint8(image_tools.resize_with_pad(blank, resize, resize))
        rgb = _b64_to_rgb_uint8(b64)
        if vflip:
            rgb = rgb[::-1]
        if hflip:
            rgb = rgb[:, ::-1]
        rgb = np.ascontiguousarray(rgb)
        return image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb, resize, resize))

    primary = names[0] if names else "table_cam"
    secondary = names[1] if len(names) > 1 else primary
    state_dim = max(_env_int("OPENPI_STATE_DIM", 8), 1)
    if _env_bool("OPENPI_ZERO_STATE", default=False):
        state = np.zeros((state_dim,), dtype=np.float32)
    else:
        state = _build_openpi_state(raw_obs, state_dim)
    prompt = str(payload.get("instruction") or raw_obs.get("instruction") or "")
    return {
        "observation/image": cam_rgb(primary),
        "observation/wrist_image": cam_rgb(secondary),
        "observation/state": state,
        "prompt": prompt,
    }


def _infer_openpi_websocket(env: Any, raw_obs: dict[str, Any], payload: dict[str, Any], ws_uri: str) -> Any:
    client = _get_openpi_websocket_client(ws_uri)
    element = _build_openpi_infer_element(env, raw_obs, payload)
    out = client.infer(element)
    if not isinstance(out, dict) or "actions" not in out:
        return []
    return out.get("actions")


def _coerce_action(action: Any, action_dim: int) -> np.ndarray:
    try:
        action_np = np.asarray(action if action is not None else [], dtype=np.float32).reshape(-1)
    except Exception:
        action_np = np.zeros((0,), dtype=np.float32)
    if action_np.size < action_dim:
        action_np = np.pad(action_np, (0, action_dim - action_np.size))
    action_np = action_np[:action_dim]
    action_np = np.nan_to_num(action_np, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(action_np, -1.0, 1.0).astype(np.float32, copy=False)


def _coerce_action_sequence(action: Any, action_dim: int) -> list[np.ndarray]:
    try:
        action_np = np.asarray(action if action is not None else [], dtype=np.float32)
    except Exception:
        return []
    if action_np.size == 0:
        return []
    if action_np.ndim == 1:
        return [_coerce_action(action_np, action_dim)]
    if action_np.ndim >= 2:
        action_np = action_np.reshape(action_np.shape[0], -1)
        return [_coerce_action(row, action_dim) for row in action_np]
    return []


def _find_key_recursively(payload: Any, key: str) -> Any:
    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        for value in payload.values():
            found = _find_key_recursively(value, key)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_key_recursively(value, key)
            if found is not None:
                return found
    return None


def _as_float_array(value: Any, min_len: int = 1) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size < int(min_len) or not np.all(np.isfinite(arr[:min_len])):
        return None
    return arr


def _extract_current_ee_pose(raw_obs: dict[str, Any]) -> dict[str, Any]:
    native_obs = (raw_obs.get("state") or {}).get("native_obs")
    eef_pos = _as_float_array(_find_key_recursively(native_obs, "eef_pos"), min_len=3)
    eef_quat = _as_float_array(_find_key_recursively(native_obs, "eef_quat"), min_len=4)
    if eef_pos is not None:
        pose = {"position": eef_pos[:3].astype(float).tolist()}
        if eef_quat is not None:
            pose["quat"] = eef_quat[:4].astype(float).tolist()
        return pose

    backend_obs = (raw_obs.get("state") or {}).get("backend_obs")
    cursor_xy = _as_float_array((backend_obs or {}).get("cursor_xy") if isinstance(backend_obs, dict) else None, min_len=2)
    if cursor_xy is not None:
        return {"position": [float(cursor_xy[0]), float(cursor_xy[1]), 0.0], "position_dim": 2}
    return {}


def _extract_gripper_pos(raw_obs: dict[str, Any]) -> np.ndarray | None:
    native_obs = (raw_obs.get("state") or {}).get("native_obs")
    for key in ("gripper_pos", "gripper_position", "gripper_joint_pos", "gripper_joint_positions"):
        candidate = _as_float_array(_find_key_recursively(native_obs, key), min_len=1)
        if candidate is not None:
            return candidate.astype(np.float32, copy=False)
    return None


def _initial_gripper_hold_action(raw_obs: dict[str, Any]) -> float:
    global _EPISODE_INITIAL_GRIPPER_ACTION
    global _EPISODE_INITIAL_GRIPPER_POS

    if _EPISODE_INITIAL_GRIPPER_ACTION is not None:
        return float(_EPISODE_INITIAL_GRIPPER_ACTION)

    override = str(os.environ.get("REMOTE_POLICY_GRIPPER_HOLD_ACTION", "")).strip()
    if override:
        try:
            action = float(override)
        except ValueError:
            action = 0.0
        _EPISODE_INITIAL_GRIPPER_ACTION = float(np.clip(action, -1.0, 1.0))
        _EPISODE_INITIAL_GRIPPER_POS = []
        return float(_EPISODE_INITIAL_GRIPPER_ACTION)

    gripper_pos = _extract_gripper_pos(raw_obs)
    if gripper_pos is not None and gripper_pos.size > 0:
        _EPISODE_INITIAL_GRIPPER_POS = gripper_pos.astype(float).tolist()
        aperture = float(np.nanmax(np.abs(gripper_pos)))
        threshold = max(_env_float("REMOTE_POLICY_GRIPPER_INITIAL_OPEN_THRESHOLD", 0.03), 0.0)
        _EPISODE_INITIAL_GRIPPER_ACTION = 1.0 if aperture >= threshold else -1.0
        return float(_EPISODE_INITIAL_GRIPPER_ACTION)

    fallback = _env_float("REMOTE_POLICY_GRIPPER_HOLD_DEFAULT_ACTION", 1.0)
    _EPISODE_INITIAL_GRIPPER_ACTION = float(np.clip(fallback, -1.0, 1.0))
    _EPISODE_INITIAL_GRIPPER_POS = []
    return float(_EPISODE_INITIAL_GRIPPER_ACTION)


def _extract_goal_error_xy(raw_obs: dict[str, Any]) -> np.ndarray | None:
    backend_obs = (raw_obs.get("state") or {}).get("backend_obs")
    if isinstance(backend_obs, dict):
        cursor_xy = _as_float_array(backend_obs.get("cursor_xy"), min_len=2)
        goal_xy = _as_float_array(backend_obs.get("goal_xy"), min_len=2)
        if cursor_xy is not None and goal_xy is not None:
            return (goal_xy[:2] - cursor_xy[:2]).astype(np.float32, copy=False)

    native_obs = (raw_obs.get("state") or {}).get("native_obs")
    for key in ("goal_error", "delta_xy", "target_delta_xy", "goal_delta_xy"):
        candidate = _as_float_array(_find_key_recursively(native_obs, key), min_len=2)
        if candidate is not None:
            return candidate[:2].astype(np.float32, copy=False)
    return None


def _clip_vector_max_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    clipped = np.asarray(vec, dtype=np.float32).copy()
    norm = float(np.linalg.norm(clipped))
    if norm > max(float(max_norm), 0.0) > 0.0:
        clipped *= float(max_norm) / max(norm, 1e-9)
    return clipped


def _apply_xy_constraints(
    action: np.ndarray,
    goal_error_xy: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    processed = action.copy()
    if processed.size < 2:
        return processed, {
            "xy_before": [],
            "xy_after": [],
            "xy_clipped": False,
            "xy_goal_direction_blocked": False,
            "xy_goal_distance_limited": False,
        }

    before = processed[:2].copy()
    max_component = max(_env_float("REMOTE_POLICY_MAX_XY_COMPONENT_M", DEFAULT_MAX_XY_COMPONENT_M), 0.0)
    if max_component > 0.0:
        processed[:2] = np.clip(processed[:2], -max_component, max_component)

    max_norm = max(_env_float("REMOTE_POLICY_MAX_XY_NORM_M", DEFAULT_MAX_XY_NORM_M), 0.0)
    if max_norm > 0.0:
        processed[:2] = _clip_vector_max_norm(processed[:2], max_norm)

    goal_direction_blocked = False
    goal_distance_limited = False
    if (
        _env_bool("REMOTE_POLICY_GOAL_ALIGNED_XY_GUARD", default=False)
        and goal_error_xy is not None
        and np.all(np.isfinite(goal_error_xy[:2]))
    ):
        goal_xy = goal_error_xy[:2].astype(np.float32, copy=False)
        goal_dist = float(np.linalg.norm(goal_xy))
        xy = processed[:2].copy()
        xy_norm = float(np.linalg.norm(xy))
        if goal_dist > 1e-6 and xy_norm > 1e-9:
            if float(np.dot(xy, goal_xy)) < 0.0:
                processed[:2] = 0.0
                goal_direction_blocked = True
            elif xy_norm > goal_dist:
                processed[:2] = xy * (goal_dist / max(xy_norm, 1e-9))
                goal_distance_limited = True

    after = processed[:2].copy()
    return processed, {
        "xy_before": before.astype(float).tolist(),
        "xy_after": after.astype(float).tolist(),
        "xy_clipped": bool(np.linalg.norm(after - before) > 1e-9),
        "xy_goal_direction_blocked": bool(goal_direction_blocked),
        "xy_goal_distance_limited": bool(goal_distance_limited),
        "xy_max_norm_m": float(max_norm),
        "xy_max_component_m": float(max_component),
    }


def _env_action_scale() -> float:
    """Metres of EE motion commanded per unit of the action we send to env.step()."""
    scale = _env_float("REMOTE_POLICY_ENV_ACTION_SCALE", DEFAULT_ENV_ACTION_SCALE)
    return scale if scale > 1e-9 else DEFAULT_ENV_ACTION_SCALE


def _workspace_bounds() -> tuple[float, float, float, float, float, float] | None:
    raw = str(os.environ.get("REMOTE_POLICY_WORKSPACE_BOUNDS", "")).strip()
    if not raw:
        return DEFAULT_WORKSPACE_BOUNDS
    if raw.lower() in {"none", "off", "disable", "disabled"}:
        return None
    try:
        values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    except Exception:
        return DEFAULT_WORKSPACE_BOUNDS
    if len(values) != 6:
        return DEFAULT_WORKSPACE_BOUNDS
    xmin, xmax, ymin, ymax, zmin, zmax = values
    if xmin > xmax or ymin > ymax or zmin > zmax:
        return DEFAULT_WORKSPACE_BOUNDS
    return values


def _apply_workspace_clip(
    action: np.ndarray,
    current_pose: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], bool]:
    processed = action.copy()
    current_pos = _as_float_array(current_pose.get("position") if isinstance(current_pose, dict) else None, min_len=3)
    if current_pos is None or processed.size < 3:
        return processed, {}, False
    position_dim = int(current_pose.get("position_dim", 3) or 3) if isinstance(current_pose, dict) else 3
    env_scale = _env_action_scale()
    target_pos = current_pos[:3] + processed[:3] * env_scale
    bounds = _workspace_bounds()
    clipped = False
    if bounds is not None:
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        clipped_target = np.asarray(
            [
                np.clip(target_pos[0], xmin, xmax),
                np.clip(target_pos[1], ymin, ymax),
                np.clip(target_pos[2], zmin, zmax) if position_dim >= 3 else target_pos[2],
            ],
            dtype=np.float32,
        )
        clipped = bool(np.linalg.norm(clipped_target - target_pos) > 1e-9)
        target_pos = clipped_target
        processed[:3] = (target_pos - current_pos[:3]) / env_scale
    target_pose = {"position": target_pos.astype(float).tolist()}
    current_quat = current_pose.get("quat") if isinstance(current_pose, dict) else None
    if current_quat is not None:
        target_pose["quat"] = current_quat
    return processed, target_pose, clipped


def _postprocess_action(action: np.ndarray, action_dim: int, raw_obs: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    processed = _coerce_action(action, action_dim).copy()
    raw_vector = processed.copy()
    action_scale = _remote_policy_action_scale()
    processed = (processed * action_scale).astype(np.float32, copy=False)
    scaled_vector = processed.copy()
    if _env_bool("REMOTE_POLICY_DISABLE_ACTION_POSTPROCESS", default=False):
        processed = np.clip(processed, -1.0, 1.0).astype(np.float32, copy=False)
        current_pose = _extract_current_ee_pose(raw_obs or {})
        debug = {
            "raw_action_vector": raw_vector.tolist(),
            "remote_action_scale": float(action_scale),
            "scaled_action_vector": scaled_vector.tolist(),
            "postprocessed_action": processed.tolist(),
            "current_ee_pose": current_pose,
            "target_pose": {},
            "delta_xyz_max_norm_m": None,
            "xyz_max_norm_disabled": True,
            "xy_constraints": {},
            "workspace_bounds": None,
            "workspace_clipped": False,
            "rotation_disabled": False,
            "gripper_binarized": False,
            "goal_error_xy": [],
            "overshoot_guard_triggered": False,
            "postprocess_disabled": True,
        }
        return processed, debug
    if action_dim >= 3:
        processed[:3] *= _env_float("REMOTE_POLICY_TRANSLATION_SCALE", 1.0)
    if action_dim >= 6:
        if _env_bool("REMOTE_POLICY_DISABLE_ROTATION", default=True):
            processed[3:6] = 0.0
        else:
            processed[3:6] *= _env_float("REMOTE_POLICY_ROTATION_SCALE", 2.0)
    if action_dim >= 7:
        mode = str(os.environ.get("REMOTE_POLICY_GRIPPER_MODE", "isaac")).strip().lower()
        raw_gripper = float(processed[-1])
        gripper_hold_initial = mode in {"hold_initial", "keep_initial", "initial", "ignore", "ignore_policy"}
        if gripper_hold_initial:
            processed[-1] = _initial_gripper_hold_action(raw_obs or {})
        elif mode in {"sign_invert", "libero", "robosuite"}:
            processed[-1] = -raw_gripper
        elif mode in {"zero_one_close", "01_close", "binary_close"}:
            processed[-1] = -1.0 if raw_gripper >= 0.5 else 1.0
        elif mode in {"zero_one_open", "01_open", "binary_open"}:
            processed[-1] = 1.0 if raw_gripper >= 0.5 else -1.0
        elif mode in {"isaac", "none", "passthrough"}:
            processed[-1] = raw_gripper
        else:
            processed[-1] = raw_gripper * _env_float("REMOTE_POLICY_GRIPPER_SCALE", 1.0)
        if _env_bool("REMOTE_POLICY_BINARIZE_GRIPPER", default=True):
            threshold = _env_float("REMOTE_POLICY_GRIPPER_BINARY_THRESHOLD", 0.0)
            processed[-1] = 1.0 if float(processed[-1]) >= threshold else -1.0

    processed = np.clip(processed, -1.0, 1.0).astype(np.float32, copy=False)
    goal_error_xy = _extract_goal_error_xy(raw_obs or {})
    processed, xy_debug = _apply_xy_constraints(processed, goal_error_xy)
    current_pose = _extract_current_ee_pose(raw_obs or {})
    processed, target_pose, workspace_clipped = _apply_workspace_clip(processed, current_pose)
    overshoot_triggered = False
    if (
        _env_bool("REMOTE_POLICY_OVERSHOOT_GUARD", default=True)
        and processed.size >= 2
        and goal_error_xy is not None
    ):
        margin = max(_env_float("REMOTE_POLICY_OVERSHOOT_MARGIN_M", 0.002), 0.0)
        current_dist = float(np.linalg.norm(goal_error_xy))
        predicted_dist = float(np.linalg.norm(goal_error_xy[:2] - processed[:2] * _env_action_scale()))
        global _EPISODE_LAST_GOAL_ERROR_XY
        previous_dist = (
            float(np.linalg.norm(_EPISODE_LAST_GOAL_ERROR_XY))
            if _EPISODE_LAST_GOAL_ERROR_XY is not None
            else current_dist
        )
        if predicted_dist > current_dist + margin or current_dist > previous_dist + margin:
            processed[:2] = 0.0
            overshoot_triggered = True
            processed, target_pose, workspace_clipped_after_guard = _apply_workspace_clip(processed, current_pose)
            workspace_clipped = bool(workspace_clipped or workspace_clipped_after_guard)
        _EPISODE_LAST_GOAL_ERROR_XY = goal_error_xy.copy()

    debug = {
        "raw_action_vector": raw_vector.tolist(),
        "remote_action_scale": float(action_scale),
        "env_action_scale": float(_env_action_scale()),
        "scaled_action_vector": scaled_vector.tolist(),
        "postprocessed_action": processed.tolist(),
        "current_ee_pose": current_pose,
        "target_pose": target_pose,
        "delta_xyz_max_norm_m": None,
        "xyz_max_norm_disabled": True,
        "xy_constraints": xy_debug,
        "workspace_bounds": list(_workspace_bounds()) if _workspace_bounds() is not None else None,
        "workspace_clipped": bool(workspace_clipped),
        "rotation_disabled": bool(_env_bool("REMOTE_POLICY_DISABLE_ROTATION", default=True)),
        "gripper_binarized": bool(action_dim >= 7 and _env_bool("REMOTE_POLICY_BINARIZE_GRIPPER", default=True)),
        "gripper_hold_initial": bool(
            action_dim >= 7
            and str(os.environ.get("REMOTE_POLICY_GRIPPER_MODE", "isaac")).strip().lower()
            in {"hold_initial", "keep_initial", "initial", "ignore", "ignore_policy"}
        ),
        "initial_gripper_action": _EPISODE_INITIAL_GRIPPER_ACTION,
        "initial_gripper_pos": list(_EPISODE_INITIAL_GRIPPER_POS),
        "goal_error_xy": goal_error_xy.tolist() if goal_error_xy is not None else [],
        "overshoot_guard_triggered": bool(overshoot_triggered),
    }
    return processed.astype(np.float32, copy=False), debug


def _refresh_processed_action_for_current_obs(
    action: np.ndarray,
    raw_obs: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if _env_bool("REMOTE_POLICY_DISABLE_ACTION_POSTPROCESS", default=False):
        processed = np.clip(
            np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0
        ).astype(np.float32, copy=False)
        current_pose = _extract_current_ee_pose(raw_obs or {})
        debug = {
            "raw_action_vector": processed.tolist(),
            "postprocessed_action": processed.tolist(),
            "current_ee_pose": current_pose,
            "target_pose": {},
            "delta_xyz_max_norm_m": None,
            "xyz_max_norm_disabled": True,
            "xy_constraints": {},
            "workspace_bounds": None,
            "workspace_clipped": False,
            "rotation_disabled": False,
            "gripper_binarized": False,
            "goal_error_xy": [],
            "overshoot_guard_triggered": False,
            "already_postprocessed": True,
            "postprocess_disabled": True,
        }
        return processed, debug
    processed = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    goal_error_xy = _extract_goal_error_xy(raw_obs or {})
    processed, xy_debug = _apply_xy_constraints(processed, goal_error_xy)
    current_pose = _extract_current_ee_pose(raw_obs or {})
    processed, target_pose, workspace_clipped = _apply_workspace_clip(processed, current_pose)
    overshoot_triggered = False
    if (
        _env_bool("REMOTE_POLICY_OVERSHOOT_GUARD", default=True)
        and processed.size >= 2
        and goal_error_xy is not None
    ):
        margin = max(_env_float("REMOTE_POLICY_OVERSHOOT_MARGIN_M", 0.002), 0.0)
        current_dist = float(np.linalg.norm(goal_error_xy))
        predicted_dist = float(np.linalg.norm(goal_error_xy[:2] - processed[:2] * _env_action_scale()))
        global _EPISODE_LAST_GOAL_ERROR_XY
        previous_dist = (
            float(np.linalg.norm(_EPISODE_LAST_GOAL_ERROR_XY))
            if _EPISODE_LAST_GOAL_ERROR_XY is not None
            else current_dist
        )
        if predicted_dist > current_dist + margin or current_dist > previous_dist + margin:
            processed[:2] = 0.0
            overshoot_triggered = True
            processed, target_pose, workspace_clipped_after_guard = _apply_workspace_clip(processed, current_pose)
            workspace_clipped = bool(workspace_clipped or workspace_clipped_after_guard)
        _EPISODE_LAST_GOAL_ERROR_XY = goal_error_xy.copy()

    debug = {
        "raw_action_vector": np.asarray(action, dtype=np.float32).reshape(-1).tolist(),
        "postprocessed_action": processed.tolist(),
        "current_ee_pose": current_pose,
        "target_pose": target_pose,
        "delta_xyz_max_norm_m": None,
        "xyz_max_norm_disabled": True,
        "xy_constraints": xy_debug,
        "workspace_bounds": list(_workspace_bounds()) if _workspace_bounds() is not None else None,
        "workspace_clipped": bool(workspace_clipped),
        "rotation_disabled": bool(_env_bool("REMOTE_POLICY_DISABLE_ROTATION", default=True)),
        "gripper_binarized": bool(processed.size >= 7 and _env_bool("REMOTE_POLICY_BINARIZE_GRIPPER", default=True)),
        "goal_error_xy": goal_error_xy.tolist() if goal_error_xy is not None else [],
        "overshoot_guard_triggered": bool(overshoot_triggered),
        "already_postprocessed": True,
    }
    return processed.astype(np.float32, copy=False), debug


def _action_postprocess_debug() -> dict[str, Any]:
    return {
        "postprocess_disabled": bool(_env_bool("REMOTE_POLICY_DISABLE_ACTION_POSTPROCESS", default=False)),
        "remote_action_scale": float(_remote_policy_action_scale()),
        "translation_scale": float(_env_float("REMOTE_POLICY_TRANSLATION_SCALE", 1.0)),
        "rotation_scale": float(_env_float("REMOTE_POLICY_ROTATION_SCALE", 2.0)),
        "max_translation_norm_m": None,
        "xyz_max_norm_disabled": True,
        "max_xy_norm_m": float(_env_float("REMOTE_POLICY_MAX_XY_NORM_M", DEFAULT_MAX_XY_NORM_M)),
        "max_xy_component_m": float(_env_float("REMOTE_POLICY_MAX_XY_COMPONENT_M", DEFAULT_MAX_XY_COMPONENT_M)),
        "goal_aligned_xy_guard": bool(_env_bool("REMOTE_POLICY_GOAL_ALIGNED_XY_GUARD", default=False)),
        "disable_rotation": bool(_env_bool("REMOTE_POLICY_DISABLE_ROTATION", default=True)),
        "gripper_mode": str(os.environ.get("REMOTE_POLICY_GRIPPER_MODE", "isaac")).strip().lower(),
        "gripper_scale": float(_env_float("REMOTE_POLICY_GRIPPER_SCALE", 1.0)),
        "binarize_gripper": bool(_env_bool("REMOTE_POLICY_BINARIZE_GRIPPER", default=True)),
        "gripper_hold_initial": str(os.environ.get("REMOTE_POLICY_GRIPPER_MODE", "isaac")).strip().lower() in {
            "hold_initial",
            "keep_initial",
            "initial",
            "ignore",
            "ignore_policy",
        },
        "initial_gripper_action": _EPISODE_INITIAL_GRIPPER_ACTION,
        "initial_gripper_pos": list(_EPISODE_INITIAL_GRIPPER_POS),
        "workspace_bounds": list(_workspace_bounds()) if _workspace_bounds() is not None else None,
        "overshoot_guard": bool(_env_bool("REMOTE_POLICY_OVERSHOOT_GUARD", default=True)),
    }


def _extract_action(resp: dict[str, Any]) -> Any:
    if "action" in resp:
        return resp.get("action")
    result = resp.get("result")
    if isinstance(result, dict) and "action" in result:
        return result.get("action")
    return []


def build_payload(env: Any, raw_obs: dict[str, Any], step: int, trace: list[dict[str, Any]]) -> dict[str, Any]:
    action_dim = _action_dim_from_env(env)
    trace_tail_k = max(_env_int("REMOTE_POLICY_TRACE_TAIL", 0), 0)
    instruction = str(raw_obs.get("instruction", ""))
    payload = {
        "step": int(step),
        "action_dim": int(action_dim),
        "action_chunk_len": int(_configured_remote_action_chunk_len()),
        "instruction": instruction,
        "observation": _to_jsonable(_safe_to_agent_obs(raw_obs)),
        "meta": _to_jsonable(raw_obs.get("meta", {})),
        "trace_tail": _to_jsonable(list(trace[-trace_tail_k:]) if trace_tail_k > 0 else []),
    }
    # Some remote VLA servers (e.g. OpenVLA forks) require explicit unnorm_key.
    unnorm_key = str(os.environ.get("REMOTE_POLICY_UNNORM_KEY", "")).strip()
    if unnorm_key:
        payload["unnorm_key"] = unnorm_key
    image_b64_by_camera = _extract_images_b64(raw_obs)
    if image_b64_by_camera:
        payload["image_b64_by_camera"] = image_b64_by_camera
        payload["images"] = [
            {"camera": camera_name, "image_b64": image_b64}
            for camera_name, image_b64 in image_b64_by_camera.items()
        ]
        # Backwards-compatible primary image for older single-image endpoints.
        for camera_name in _camera_names_from_env():
            image_b64 = image_b64_by_camera.get(camera_name, "")
            if image_b64:
                payload["image_b64"] = image_b64
                payload["camera_name"] = camera_name
                break
    return payload


def policy(env: Any, raw_obs: dict[str, Any], step: int, trace: list[dict[str, Any]]) -> np.ndarray:
    global _EPISODE_CACHE_HIT_COUNT
    global _EPISODE_HTTP_REQUEST_COUNT
    global _EPISODE_TOTAL_EFFECTIVE_CHUNK_LEN
    global _EPISODE_TOTAL_SERVER_CHUNK_LEN
    global _EPISODE_LAST_ACTION
    global _LAST_POLICY_DEBUG
    global _LAST_EFFECTIVE_CHUNK_LEN
    global _LAST_SERVER_CHUNK_LEN

    if int(step) == 0 and len(trace) == 0:
        _reset_episode_stats(env)
    elif not _EPISODE_ACTIVE:
        _reset_episode_stats(env)

    inter_action_sleep_s = max(_env_float("REMOTE_POLICY_INTER_ACTION_SLEEP_S", 0.0), 0.0)
    if int(step) > 0 and inter_action_sleep_s > 0.0:
        time.sleep(inter_action_sleep_s)
    debug_action = _env_bool("REMOTE_POLICY_DEBUG_ACTION", default=False)

    if _ACTION_CHUNK_CACHE:
        _EPISODE_CACHE_HIT_COUNT += 1
        raw_action_from_cache = _ACTION_CHUNK_CACHE.pop(0)
        action_from_cache, adapter_debug = _postprocess_action(raw_action_from_cache, action_dim=_action_dim_from_env(env), raw_obs=raw_obs)
        _EPISODE_LAST_ACTION = action_from_cache.copy()
        _LAST_POLICY_DEBUG = {
            "episode_id": str(_EPISODE_ID),
            "step": int(step),
            "request_id": "",
            "raw_action": raw_action_from_cache.tolist(),
            "raw_action_from_policy": raw_action_from_cache.tolist(),
            "raw_action_shape": _shape_of(raw_action_from_cache),
            "postprocessed_action": _EPISODE_LAST_ACTION.tolist(),
            "sent_action_to_env": _EPISODE_LAST_ACTION.tolist(),
            "sent_action_shape": _shape_of(_EPISODE_LAST_ACTION),
            "current_ee_pose": adapter_debug.get("current_ee_pose", {}),
            "target_pose": adapter_debug.get("target_pose", {}),
            "chunk_len": int(max(_LAST_EFFECTIVE_CHUNK_LEN, 1)),
            "server_chunk_len": int(max(_LAST_SERVER_CHUNK_LEN, 0)),
            "action_source": "cache",
            "from_cache": True,
            "from_hold": False,
            "cache_size_after_pop": int(len(_ACTION_CHUNK_CACHE)),
            "payload_image_cameras": [],
            "action_postprocess": _action_postprocess_debug(),
            "adapter_debug": adapter_debug,
        }
        if debug_action:
            print(
                "[geomind_client] action_recv "
                + json.dumps(
                    {
                        "step": int(step),
                        "request_id": "",
                        "source": "cache",
                        "raw_action": raw_action_from_cache.tolist(),
                        "raw_action_shape": _shape_of(raw_action_from_cache),
                        "sent_action": _EPISODE_LAST_ACTION.tolist(),
                        "sent_action_shape": _shape_of(_EPISODE_LAST_ACTION),
                        "current_ee_pose": adapter_debug.get("current_ee_pose", {}),
                        "target_pose": adapter_debug.get("target_pose", {}),
                        "chunk_len": int(max(_LAST_EFFECTIVE_CHUNK_LEN, 1)),
                        "server_chunk_len": int(max(_LAST_SERVER_CHUNK_LEN, 0)),
                        "cache_size_after_pop": int(len(_ACTION_CHUNK_CACHE)),
                        "adapter_debug": adapter_debug,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        return _EPISODE_LAST_ACTION

    action_dim = _action_dim_from_env(env)
    use_action_chunk = _env_bool("REMOTE_POLICY_USE_ACTION_CHUNK", default=False)
    configured_chunk_len = _configured_remote_action_chunk_len()
    policy_hz = _configured_policy_hz()
    request_interval = 1
    if policy_hz > 0.0 and _EPISODE_LAST_ENV_STEP_HZ > policy_hz:
        request_interval = max(int(round(_EPISODE_LAST_ENV_STEP_HZ / policy_hz)), 1)
    if (
        not use_action_chunk
        and request_interval > 1
        and int(step) > 0
        and int(step) % request_interval != 0
        and _EPISODE_LAST_ACTION is not None
    ):
        held_action, adapter_debug = _refresh_processed_action_for_current_obs(_EPISODE_LAST_ACTION, raw_obs=raw_obs)
        _EPISODE_LAST_ACTION = held_action.copy()
        _LAST_POLICY_DEBUG = {
            "episode_id": str(_EPISODE_ID),
            "step": int(step),
            "request_id": "",
            "raw_action": held_action.tolist(),
            "raw_action_from_policy": [],
            "raw_action_shape": [],
            "postprocessed_action": held_action.tolist(),
            "sent_action_to_env": held_action.tolist(),
            "sent_action_shape": _shape_of(held_action),
            "current_ee_pose": adapter_debug.get("current_ee_pose", {}),
            "target_pose": adapter_debug.get("target_pose", {}),
            "chunk_len": 1,
            "server_chunk_len": 0,
            "action_source": "hold",
            "from_cache": False,
            "from_hold": True,
            "hold_request_interval": int(request_interval),
            "cache_size_after_pop": int(len(_ACTION_CHUNK_CACHE)),
            "payload_image_cameras": [],
            "action_postprocess": _action_postprocess_debug(),
            "adapter_debug": adapter_debug,
        }
        if debug_action:
            print(
                "[geomind_client] action_recv "
                + json.dumps(
                    {
                        "step": int(step),
                        "request_id": "",
                        "source": "hold",
                        "sent_action": held_action.tolist(),
                        "sent_action_shape": _shape_of(held_action),
                        "hold_request_interval": int(request_interval),
                        "current_ee_pose": adapter_debug.get("current_ee_pose", {}),
                        "target_pose": adapter_debug.get("target_pose", {}),
                        "adapter_debug": adapter_debug,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        return _EPISODE_LAST_ACTION

    remote_url = str(os.environ.get("REMOTE_POLICY_URL", DEFAULT_REMOTE_POLICY_URL)).strip()
    timeout_s = _env_float("REMOTE_POLICY_TIMEOUT_S", 10.0)
    timing = _env_bool("REMOTE_POLICY_TIMING")
    t0 = time.perf_counter()
    payload = build_payload(env, raw_obs, step, trace)
    payload_image_cameras = sorted((payload.get("image_b64_by_camera") or {}).keys())
    t_payload = time.perf_counter()

    action: Any = []
    request_id = f"{_EPISODE_ID}_{int(step)}"
    try:
        openpi_ws = _openpi_websocket_uri_for_policy()
        if openpi_ws:
            action = _infer_openpi_websocket(env, raw_obs, payload, openpi_ws)
            request_id = f"{_EPISODE_ID}_{int(step)}"
        else:
            resp = _post_json(remote_url, payload=payload, timeout_s=timeout_s)
            request_id = _extract_request_id(resp, request_id)
            action = _extract_action(resp)
    except (error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        if _env_bool("REMOTE_POLICY_DEBUG"):
            print(f"[geomind_client] remote policy request failed: {exc}")
    except Exception as exc:
        if _env_bool("REMOTE_POLICY_DEBUG"):
            print(f"[geomind_client] unexpected remote policy error: {exc}")

    t_done = time.perf_counter()
    sequence = _coerce_action_sequence(action, action_dim)
    server_chunk_len = int(len(sequence))
    raw_sequence = sequence if sequence else [_coerce_action([], action_dim)]
    if use_action_chunk:
        if len(raw_sequence) == 1 and configured_chunk_len > 1:
            base = raw_sequence[0]
            effective_raw_sequence = [base.copy() for _ in range(configured_chunk_len)]
        else:
            effective_raw_sequence = raw_sequence[:configured_chunk_len]
    else:
        effective_raw_sequence = [raw_sequence[0]]
    first_action, adapter_debug = _postprocess_action(effective_raw_sequence[0], action_dim, raw_obs=raw_obs)
    effective_sequence = [first_action]
    _EPISODE_HTTP_REQUEST_COUNT += 1
    _EPISODE_TOTAL_EFFECTIVE_CHUNK_LEN += int(len(effective_raw_sequence))
    _EPISODE_TOTAL_SERVER_CHUNK_LEN += int(server_chunk_len)
    _LAST_EFFECTIVE_CHUNK_LEN = int(len(effective_raw_sequence))
    _LAST_SERVER_CHUNK_LEN = int(server_chunk_len)
    if use_action_chunk and len(effective_raw_sequence) > 1:
        _ACTION_CHUNK_CACHE.extend(action.copy() for action in effective_raw_sequence[1:])
    if timing:
        stats = get_remote_policy_stats()
        print(
            "[geomind_client] timing "
            f"step={int(step)} payload_s={t_payload - t0:.3f} "
            f"request_s={t_done - t_payload:.3f} "
            f"server_chunk={server_chunk_len} executed_chunk={len(effective_raw_sequence)} "
            f"cache={len(_ACTION_CHUNK_CACHE)} env_step_hz={stats['env_step_hz']:.3f} "
            f"physics_step_hz={stats['physics_step_hz']:.3f} action_repeat={stats['action_repeat']} "
            f"remote_action_chunk_len={stats['remote_action_chunk_len']} "
            f"estimated_remote_policy_hz={stats['estimated_remote_policy_hz']:.3f} "
            f"http_requests={stats['episode_http_request_count']} "
            f"avg_chunk_len={stats['average_effective_chunk_length']:.3f} "
            f"avg_server_chunk_len={stats['average_server_chunk_length']:.3f} "
            f"cache_hits={stats['cache_hit_count']}",
            flush=True,
        )
    _EPISODE_LAST_ACTION = effective_sequence[0].copy()
    _LAST_POLICY_DEBUG = {
        "episode_id": str(_EPISODE_ID),
        "step": int(step),
        "request_id": str(request_id),
        "raw_action": _to_jsonable(action),
        "raw_action_from_policy": _to_jsonable(action),
        "raw_action_shape": _shape_of(action),
        "postprocessed_action": _EPISODE_LAST_ACTION.tolist(),
        "sent_action_to_env": _EPISODE_LAST_ACTION.tolist(),
        "sent_action_shape": _shape_of(_EPISODE_LAST_ACTION),
        "current_ee_pose": adapter_debug.get("current_ee_pose", {}),
        "target_pose": adapter_debug.get("target_pose", {}),
        "chunk_len": int(len(effective_raw_sequence)),
        "server_chunk_len": int(server_chunk_len),
        "action_source": "request",
        "from_cache": False,
        "from_hold": False,
        "cache_size_after_pop": int(len(_ACTION_CHUNK_CACHE)),
        "payload_image_cameras": payload_image_cameras,
        "action_postprocess": _action_postprocess_debug(),
        "adapter_debug": adapter_debug,
    }
    if debug_action:
        print(
            "[geomind_client] action_recv "
            + json.dumps(
                {
                    "step": int(step),
                    "request_id": str(request_id),
                    "source": "request",
                    "raw_action": _to_jsonable(action),
                    "raw_action_shape": _shape_of(action),
                    "sent_action": _EPISODE_LAST_ACTION.tolist(),
                    "sent_action_shape": _shape_of(_EPISODE_LAST_ACTION),
                    "current_ee_pose": adapter_debug.get("current_ee_pose", {}),
                    "target_pose": adapter_debug.get("target_pose", {}),
                    "chunk_len": int(len(effective_raw_sequence)),
                    "server_chunk_len": int(server_chunk_len),
                    "payload_image_cameras": payload_image_cameras,
                    "action_postprocess": _action_postprocess_debug(),
                    "adapter_debug": adapter_debug,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return _EPISODE_LAST_ACTION
