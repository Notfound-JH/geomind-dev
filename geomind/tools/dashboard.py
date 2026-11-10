"""Local, live GeoMind results browser. Cleanup always requires an explicit confirmation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import secrets
import shutil
import threading
import time

from geomind.tools.instruction_zh import translate_instruction
from urllib.parse import unquote, urlsplit


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, fallback=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return fallback


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


class Dashboard:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.outputs = self.root / 'outputs'
        self.configs = self.root / 'geomind/configs'
        self.history_path = self.root / 'data/dashboard/history.json'
        self.history = read_json(self.history_path, {'runs': {}, 'changes': [], 'config_hashes': {}})
        self.csrf = secrets.token_urlsafe(32)
        self.plans = {}
        self.lock = threading.RLock()

    def inventory(self):
        hashes, groups, errors = {}, {}, []
        for group in ('task_definitions', 'tasks', 'vla_tasks', 'episodes', 'scenes', 'layouts', 'protocols'):
            items = []
            for path in sorted((self.configs / group).glob('*.json')):
                if path.is_symlink():
                    continue
                payload = read_json(path)
                if not isinstance(payload, dict):
                    errors.append(str(path.relative_to(self.root)))
                    continue
                hashes[str(path.relative_to(self.configs))] = hashlib.sha256(path.read_bytes()).hexdigest()
                items.append(payload)
            groups[group] = items
        # Track implementation and documentation changes as well as visible configuration.
        extra = list((self.root / 'geomind').rglob('*.py')) + list((self.root / 'geomind/web').glob('*.html'))
        extra += list((self.root / 'geomind_lab').rglob('*.py')) + list((self.root / 'docs').glob('*.md'))
        extra += [self.root / 'README.md']
        for path in extra:
            if path.is_file() and not path.is_symlink():
                hashes['source/' + str(path.relative_to(self.root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        old = self.history['config_hashes']
        if old != hashes:
            if old:
                for key in sorted(old.keys() | hashes.keys()):
                    if key.startswith('source/') and not any(k.startswith('source/') for k in old):
                        continue
                    if old.get(key) != hashes.get(key):
                        action = '新增' if key not in old else '删除' if key not in hashes else '修改'
                        self.history['changes'].insert(0, {'at': now(), 'kind': 'config', 'title': f'{action} {key}'})
            else:
                self.history['changes'].insert(0, {'at': now(), 'kind': 'config', 'title': f'首次索引 {len(hashes)} 个配置文件'})
            self.history['config_hashes'] = hashes
            self.persist()
        return groups, errors

    def persist(self):
        self.history['changes'] = self.history['changes'][:200]
        save_json(self.history_path, self.history)

    def path_url(self, value):
        if not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = self.root / path
        try:
            relative = path.resolve().relative_to(self.outputs.resolve())
            if path.is_file():
                return '/media/' + relative.as_posix()
        except (ValueError, OSError):
            pass
        return None

    @staticmethod
    def is_run(path):
        return (path / 'report.json').is_file() or ((path / 'cases').is_dir() and (path / 'traces').is_dir())

    def run_summary(self, path):
        report = read_json(path / 'report.json', {}) or {}
        if not report:
            report = {'results': [v for p in sorted((path / 'cases').glob('*.json')) if isinstance((v := read_json(p)), dict)]}
        cases = []
        for case in report.get('results', []):
            video = case.get('video') or {}
            case_id = case['episode_id'] + ('_negative' if case.get('negative_control') else '')
            scene = case.get('scene_provenance') or {}
            cases.append({'id': case_id, 'task': case.get('task_id'), 'domain': case.get('domain'),
                'release_version': case.get('release_version') or ('2.0' if report.get('backend') == 'isaaclab_physx' else None),
                'success': case.get('success'), 'verified': case.get('verification_passed'),
                'negative': case.get('negative_control', False), 'instruction': case.get('instruction', ''),
                'instruction_zh': case.get('instruction_zh') or translate_instruction(case.get('instruction', '')),
                'error_label': case.get('error_label', 'TCP 误差'),
                'error_mm': ((case.get('physical') or {}).get('position_error_m', 0) or 0) * 1000,
                'scene': f"scene {scene['scene_id']}" if scene.get('scene_id') else scene.get('mode', '场景未记录'),
                'objects': [a['asset'] for a in case.get('object_assets', [])],
                'camera_videos': [{'name': v['name'], 'url': self.path_url(v['path'])} for v in case.get('camera_videos', [])],
                'observer': self.path_url(video.get('observer')), 'agent': self.path_url(video.get('agent')),
                'poster': self.path_url(path / 'videos' / (case_id + '_first.png'))})
        files, size = self.fingerprint_files(path.name)
        stat = ((path / 'report.json') if (path / 'report.json').is_file() else path).stat()
        return {'id': path.name, 'at': datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            'verified': report.get('physical_execution_verified', False),
            'failure': report.get('failure'), 'backend': report.get('backend', '未记录'),
            'policy': report.get('policy', '未记录'), 'cases': cases,
            'positive': report.get('positive_cases', 0), 'passed': report.get('positive_successes', 0),
            'video_count': sum(len([v for v in c['camera_videos'] if v['url']]) if c['camera_videos'] else sum(bool(c[k]) for k in ('observer', 'agent')) for c in cases),
            'bytes': size, 'file_count': len(files), 'deleted': False,
            'report': self.path_url(path / 'report.json')}

    def targets(self, identity):
        if not identity or Path(identity).name != identity or identity in ('.', '..'):
            raise ValueError('无效的结果目录')
        path = self.outputs / identity
        if path.is_symlink() or not path.is_dir() or not self.is_run(path):
            raise ValueError('只允许清理含报告或 cases/traces 的测试目录')
        if path.resolve().parent != self.outputs.resolve():
            raise ValueError('结果目录越界')
        return [p for p in (path, self.outputs / (identity + '.zip'), self.outputs / (identity + '.zip.sha256')) if p.exists() or p.is_symlink()]

    def fingerprint_files(self, identity):
        records = []
        for target in self.targets(identity):
            if target.is_symlink():
                raise ValueError('结果中含符号链接，拒绝自动删除')
            paths = [target] if target.is_file() else [target, *sorted(target.rglob('*'))]
            for path in paths:
                if path.is_symlink():
                    raise ValueError('结果中含符号链接，拒绝自动删除')
                if path.is_file():
                    stat = path.stat()
                    records.append([str(path.relative_to(self.outputs)), stat.st_size, stat.st_mtime_ns])
        return records, sum(r[1] for r in records)

    def state(self):
        with self.lock:
            groups, errors = self.inventory()
            runs = []
            self.outputs.mkdir(exist_ok=True)
            for path in sorted(self.outputs.iterdir()):
                if path.is_symlink() or not path.is_dir() or not self.is_run(path):
                    continue
                try:
                    run = self.run_summary(path)
                    runs.append(run)
                    previous = self.history['runs'].get(path.name)
                    if previous is None or previous.get('at') != run['at'] or previous.get('deleted'):
                        self.history['runs'][path.name] = run
                        self.history['changes'].insert(0, {'at': now(), 'kind': 'run', 'title': f'发现测试结果 {path.name}'})
                        self.persist()
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    errors.append(f'{path.name}: {exc}')
            current = {r['id'] for r in runs}
            for key, summary in self.history['runs'].items():
                if key not in current:
                    summary = dict(summary)
                    summary['cases'] = [dict(c, instruction_zh=c.get('instruction_zh') or translate_instruction(c.get('instruction', '')),
                        release_version=c.get('release_version') or ('2.0' if summary.get('backend') == 'isaaclab_physx' else None)) for c in summary['cases']]
                    summary['missing'] = not summary.get('deleted')
                    summary['bytes'] = 0
                    summary['report'] = None
                    summary['cases'] = [dict(c, observer=None, agent=None, poster=None, camera_videos=[]) for c in summary['cases']]
                    runs.append(summary)
            runs.sort(key=lambda r: r['at'], reverse=True)
            return {'updated_at': now(), 'csrf': self.csrf, 'runs': runs, 'config': groups,
                    'changes': self.history['changes'], 'errors': errors}

    def plan(self, identity):
        with self.lock:
            summary = self.run_summary(self.outputs / identity) if Path(identity).name == identity else None
            if summary is None:
                raise ValueError('无效目录')
            records, size = self.fingerprint_files(identity)
            token = secrets.token_urlsafe(24)
            plan = {'token': token, 'id': identity, 'files': records, 'bytes': size,
                    'expires': time.time() + 300, 'summary': summary}
            self.plans = {k: v for k, v in self.plans.items() if v['expires'] > time.time()}
            self.plans[token] = plan
            return {'token': token, 'id': identity, 'bytes': size, 'file_count': len(records),
                    'targets': [str(p.relative_to(self.root)) for p in self.targets(identity)],
                    'files': [r[0] for r in records], 'expires_in_s': 300}

    def delete(self, token, confirmation):
        with self.lock:
            plan = self.plans.pop(token, None)
            if not plan or plan['expires'] < time.time():
                raise ValueError('清理预览已过期，请重新预览')
            if confirmation != plan['id']:
                raise ValueError('确认文字必须与目录名完全一致')
            current, _ = self.fingerprint_files(plan['id'])
            if current != plan['files']:
                raise ValueError('文件在预览后发生变化，已取消删除，请重新预览')
            summary = dict(plan['summary'], deleted=True, deleted_at=now(), bytes=0)
            summary['cases'] = [dict(c, observer=None, agent=None, poster=None, camera_videos=[]) for c in summary['cases']]
            summary['report'] = None
            # Persist the compact verification summary before removing any raw artifacts.
            self.history['runs'][plan['id']] = summary
            self.persist()
            for path in self.targets(plan['id']):
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            self.history['changes'].insert(0, {'at': now(), 'kind': 'cleanup', 'title': f"已确认清理 {plan['id']}"})
            self.persist()
            return {'deleted': plan['id'], 'freed_bytes': plan['bytes']}


class Handler(BaseHTTPRequestHandler):
    def json_response(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def valid_host(self):
        return self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}

    def do_GET(self):
        if not self.valid_host():
            return self.json_response({'error': '仅允许本机访问'}, 403)
        path = unquote(urlsplit(self.path).path)
        if path == '/api/state':
            return self.json_response(self.server.dashboard.state())
        if path == '/':
            file = Path(__file__).resolve().parents[1] / 'web/dashboard.html'
        elif path.startswith('/media/'):
            base = self.server.dashboard.outputs.resolve()
            file = (base / path[len('/media/'):]).resolve()
            if not file.is_relative_to(base) or file.suffix not in {'.mp4', '.png', '.jpg', '.json', '.zip'}:
                return self.send_error(403)
        else:
            return self.send_error(404)
        if not file.is_file():
            return self.send_error(404)
        length = file.stat().st_size
        start, end = 0, length - 1
        range_header = self.headers.get('Range')
        try:
            if range_header:
                if not range_header.startswith('bytes=') or ',' in range_header:
                    raise ValueError()
                a, b = range_header[6:].split('-', 1)
                if a:
                    start = int(a)
                    end = min(int(b), end) if b else end
                else:
                    start = max(0, length-int(b))
                if start < 0 or start > end or start >= length:
                    raise ValueError()
        except ValueError:
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{length}')
            self.end_headers()
            return
        self.send_response(206 if range_header else 200)
        self.send_header('Content-Type', mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(end-start+1))
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if range_header:
            self.send_header('Content-Range', f'bytes {start}-{end}/{length}')
        self.end_headers()
        try:
            with file.open('rb') as stream:
                stream.seek(start)
                remaining = end-start+1
                while remaining:
                    chunk = stream.read(min(remaining, 1024*1024))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        dashboard = self.server.dashboard
        if (not self.valid_host() or self.headers.get('Origin') != 'http://' + self.headers.get('Host', '')
                or self.headers.get('X-CSRF-Token') != dashboard.csrf):
            return self.json_response({'error': '请求来源校验失败'}, 403)
        try:
            length = int(self.headers.get('Content-Length', 0))
            if not 0 < length < 8192:
                raise ValueError('无效请求长度')
            body = json.loads(self.rfile.read(length))
            if self.path == '/api/cleanup/plan':
                result = dashboard.plan(body['id'])
            elif self.path == '/api/cleanup/confirm':
                result = dashboard.delete(body['token'], body['confirmation'])
            else:
                return self.json_response({'error': '不存在的操作'}, 404)
            self.json_response(result)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self.json_response({'error': str(exc)}, 400)

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.dashboard = Dashboard(args.root)
    print(f'GeoMind 展示页：http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == '__main__':
    main()
