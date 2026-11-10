"""Exercise real localhost HTTP downloads using a small temporary ZIP only."""
from __future__ import annotations

from functools import partial
import hashlib
from http.server import ThreadingHTTPServer
import importlib.util
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('geomind_delivery_server', ROOT / 'scripts/serve_delivery.py')
SERVER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVER_MODULE)


class QuietDeliveryHandler(SERVER_MODULE.DeliveryHandler):
    def log_message(self, *args):
        pass


class DeliveryHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='geomind-delivery-http-')
        cls.area = Path(cls.temporary.name)
        cls.release = cls.area / 'release'
        cls.release.mkdir()
        with zipfile.ZipFile(cls.release / 'fixture.zip', 'w', compression=zipfile.ZIP_DEFLATED,
                             allowZip64=True) as archive:
            archive.writestr('sample.bin', bytes(range(256)) * 4096)
        cls.payload = (cls.release / 'fixture.zip').read_bytes()
        (cls.release / 'empty.bin').write_bytes(b'')
        (cls.area / 'outside.zip').write_bytes(b'outside the allowed release')
        cls.symlink_available = True
        try:
            (cls.release / 'escape.zip').symlink_to(cls.area / 'outside.zip')
        except (OSError, NotImplementedError):
            cls.symlink_available = False
        try:
            cls.server = ThreadingHTTPServer(
                ('127.0.0.1', 0), partial(QuietDeliveryHandler, directory=cls.release))
        except PermissionError as exc:
            cls.temporary.cleanup()
            raise unittest.SkipTest('This environment disallows binding a localhost test socket') from exc
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_address[1]}'
        # These requests are exclusively to the fixture's loopback socket.
        cls.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls.temporary.cleanup()

    def request(self, path='/fixture.zip', *, headers=None, method='GET'):
        request = urllib.request.Request(self.url + path, headers=headers or {}, method=method)
        try:
            with self.opener.open(request, timeout=5) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as response:
            with response:
                return response.code, dict(response.headers), response.read()

    def assert_invalid_range(self, value):
        status, headers, body = self.request(headers={'Range': value})
        self.assertEqual(status, 416)
        self.assertEqual(headers['Content-Range'], f'bytes */{len(self.payload)}')
        self.assertEqual(headers['Content-Length'], '0')
        self.assertEqual(body, b'')

    def test_full_get_content_hash(self):
        status, _, body = self.request()
        self.assertEqual(status, 200)
        self.assertEqual(hashlib.sha256(body).digest(), hashlib.sha256(self.payload).digest())

    def test_zip_download_headers(self):
        status, headers, _ = self.request()
        self.assertEqual(status, 200)
        self.assertEqual(headers['Content-Type'], 'application/zip')
        self.assertTrue(headers['Content-Disposition'].startswith('attachment;'))
        self.assertEqual(headers['Accept-Ranges'], 'bytes')

    def test_head_returns_length_without_body(self):
        status, headers, body = self.request(method='HEAD')
        self.assertEqual(status, 200)
        self.assertEqual(int(headers['Content-Length']), len(self.payload))
        self.assertEqual(body, b'')

    def test_two_resumed_ranges_reconstruct_zip_hash(self):
        cut = len(self.payload) // 2
        first = self.request(headers={'Range': f'bytes=0-{cut-1}'})
        second = self.request(headers={'Range': f'bytes={cut}-'})
        self.assertEqual((first[0], second[0]), (206, 206))
        self.assertEqual(hashlib.sha256(first[2] + second[2]).digest(),
                         hashlib.sha256(self.payload).digest())

    def test_range_response_headers(self):
        cut = len(self.payload) // 2
        status, headers, body = self.request(headers={'Range': f'bytes=0-{cut-1}'})
        self.assertEqual(status, 206)
        self.assertEqual(headers['Content-Range'], f'bytes 0-{cut-1}/{len(self.payload)}')
        self.assertEqual(int(headers['Content-Length']), len(body))

    def test_suffix_range(self):
        status, _, body = self.request(headers={'Range': 'bytes=-17'})
        self.assertEqual(status, 206)
        self.assertEqual(body, self.payload[-17:])

    def test_oversized_end_is_clamped(self):
        status, _, body = self.request(headers={'Range': 'bytes=0-9999999999'})
        self.assertEqual(status, 206)
        self.assertEqual(body, self.payload)

    def test_head_with_range(self):
        status, headers, body = self.request(headers={'Range': 'bytes=5-19'}, method='HEAD')
        self.assertEqual(status, 206)
        self.assertEqual(headers['Content-Length'], '15')
        self.assertEqual(body, b'')

    def test_unsatisfiable_start(self):
        self.assert_invalid_range(f'bytes={len(self.payload)}-')

    def test_backwards_range(self):
        self.assert_invalid_range('bytes=9-2')

    def test_zero_suffix(self):
        self.assert_invalid_range('bytes=-0')

    def test_empty_range(self):
        self.assert_invalid_range('bytes=-')

    def test_multiple_ranges_are_rejected(self):
        self.assert_invalid_range('bytes=1-2,4-5')

    def test_non_byte_unit_is_rejected(self):
        self.assert_invalid_range('items=0-1')

    def test_non_numeric_range_is_rejected(self):
        self.assert_invalid_range('bytes=abc-def')

    def test_empty_file_has_no_satisfiable_range(self):
        status, headers, body = self.request('/empty.bin', headers={'Range': 'bytes=0-0'})
        self.assertEqual(status, 416)
        self.assertEqual(headers['Content-Range'], 'bytes */0')
        self.assertEqual(body, b'')

    def test_if_range_matching_etag(self):
        _, initial, _ = self.request(method='HEAD')
        status, _, body = self.request(headers={'Range': 'bytes=0-4', 'If-Range': initial['ETag']})
        self.assertEqual(status, 206)
        self.assertEqual(body, self.payload[:5])

    def test_if_range_mismatch_returns_full_file(self):
        status, _, body = self.request(headers={'Range': 'bytes=0-4', 'If-Range': '"mismatch"'})
        self.assertEqual(status, 200)
        self.assertEqual(body, self.payload)

    def test_encoded_parent_traversal_is_forbidden(self):
        self.assertEqual(self.request('/%2e%2e/outside.zip')[0], 403)

    def test_symlink_outside_release_is_forbidden(self):
        if not self.symlink_available:
            self.skipTest('This platform cannot create the symlink fixture')
        self.assertEqual(self.request('/escape.zip')[0], 403)

    def test_backslash_traversal_is_forbidden(self):
        self.assertEqual(self.request('/..%5coutside.zip')[0], 403)

    def test_missing_file(self):
        self.assertEqual(self.request('/missing.zip')[0], 404)

    def test_directory_listing_is_not_exposed(self):
        self.assertEqual(self.request('/')[0], 404)


if __name__ == '__main__':
    unittest.main()
