#!/usr/bin/env python3
"""Serve one release directory with resumable downloads on localhost.

Does not list directories or follow symlinks outside --directory. Use SSH port
forwarding to reach a remote server; default binding is 127.0.0.1 only.
"""
from __future__ import annotations

import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
from pathlib import Path
import re
from urllib.parse import quote, unquote, urlsplit
from email.utils import formatdate, parsedate_to_datetime


def byte_range(value: str, size: int) -> tuple[int, int]:
    """Parse one inclusive byte range; invalid/unsatisfiable means ValueError."""
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', value.strip(), flags=re.IGNORECASE)
    if match is None or size <= 0:
        raise ValueError('Invalid or unsatisfiable single range')
    left, right = match.groups()
    if not left:
        if not right or int(right) <= 0:
            raise ValueError('Empty or zero suffix range')
        return max(0, size - int(right)), size - 1
    start = int(left)
    end = min(int(right), size - 1) if right else size - 1
    if start >= size or start > end:
        raise ValueError('Range is outside the file')
    return start, end


class DeliveryHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'GeoMindDelivery/1.0'

    def __init__(self, *args, directory: Path, **kwargs):
        self.release_root = Path(directory).resolve()
        super().__init__(*args, **kwargs)

    def do_HEAD(self):
        self._serve(head=True)

    def do_GET(self):
        self._serve(head=False)

    def _path(self) -> Path:
        parsed = urlsplit(self.path)
        if parsed.scheme or parsed.netloc:
            raise PermissionError('Absolute URLs are not served')
        decoded = unquote(parsed.path, errors='strict')
        if '\x00' in decoded or '\\' in decoded:
            raise PermissionError('Invalid path')
        relative = Path(decoded.lstrip('/'))
        if '..' in relative.parts:
            raise PermissionError('Path traversal is not served')
        target = (self.release_root / relative).resolve()
        if not target.is_relative_to(self.release_root):
            raise PermissionError('Outside release directory')
        if target.is_dir():
            target = (target / 'index.html').resolve()
            if not target.is_relative_to(self.release_root):
                raise PermissionError('Outside release directory')
        return target

    def _serve(self, *, head: bool):
        try:
            path = self._path()
            if not path.is_file():
                self.send_error(404, 'Release file not found')
                return
            stat = path.stat()
            size = stat.st_size
        except (PermissionError, UnicodeError, ValueError):
            self.send_error(403, 'Only files inside the release directory are served')
            return
        except OSError:
            self.send_error(404, 'Release file not found')
            return

        etag = f'"{stat.st_size:x}-{stat.st_mtime_ns:x}"'
        requested_range = self.headers.get('Range')
        if requested_range and self.headers.get('If-Range'):
            condition = self.headers['If-Range'].strip()
            matches = condition == etag
            if not matches:
                try:
                    matches = int(parsedate_to_datetime(condition).timestamp()) >= int(stat.st_mtime)
                except (ValueError, TypeError, OverflowError):
                    pass
            if not matches:
                requested_range = None
        start, end = 0, size - 1
        if requested_range:
            try:
                start, end = byte_range(requested_range, size)
            except ValueError:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
        length = end - start + 1 if size else 0
        try:
            handle = path.open('rb') if not head else None
        except OSError:
            self.send_error(403, 'Cannot read release file')
            return

        self.send_response(206 if requested_range else 200)
        self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        self.send_header('Content-Length', str(length))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Last-Modified', formatdate(stat.st_mtime, usegmt=True))
        self.send_header('ETag', etag)
        if requested_range:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        if path.suffix.lower() == '.zip':
            ascii_name = ''.join(c if 32 <= ord(c) < 127 and c not in '\\"' else '_' for c in path.name)
            self.send_header('Content-Disposition', f'attachment; filename="{ascii_name}"; filename*=UTF-8\'\'{quote(path.name)}')
        self.end_headers()
        if head:
            return
        try:
            with handle:
                handle.seek(start)
                remaining = length
                while remaining:
                    block = handle.read(min(1024 * 1024, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError):
            return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True, help='Only this completed release directory is served')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8782)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_dir():
        parser.error('--directory must be an existing release directory')
    if not 0 <= args.port <= 65535:
        parser.error('--port must be in 0..65535')
    handler = partial(DeliveryHandler, directory=directory)
    with ThreadingHTTPServer((args.host, args.port), handler) as server:
        host, port = server.server_address[:2]
        print(f'Serving completed release at http://{host}:{port}/ from {directory}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
