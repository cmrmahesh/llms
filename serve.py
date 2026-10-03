#!/usr/bin/env python3
"""Serve this folder on http://localhost:8000 with caching disabled.

ES modules and the Web Worker need to be served over HTTP (not file://).
Usage: python3 serve.py [port]
"""
import http.server
import sys


class NoCache(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, '.wasm': 'application/wasm', '.js': 'text/javascript'}

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f'Serving on http://localhost:{port}')
    http.server.ThreadingHTTPServer(('127.0.0.1', port), NoCache).serve_forever()
