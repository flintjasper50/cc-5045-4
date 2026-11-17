"""Stdlib HTTP server: real REST API + static frontend hosting.

Endpoints (all computation happens here, nothing is hardcoded):
  GET  /api/scenes              list deterministic scenes
  GET  /api/devices             honest device/GPU discovery + coverage note
  POST /api/render              run selected devices on a scene, compare
  GET  /api/config/<scene>      reproducibility configuration + sha256
  GET  /                        frontend single-page panel
  GET  /static/<f>              frontend assets
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from raster import available_scenes, render_job, reproducibility_config
from raster.devices import discover_devices

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "frontend")

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".png": "image/png",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "MultiRenderVerify/1.0"

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, rel_path):
        path = os.path.normpath(os.path.join(FRONTEND_DIR, rel_path))
        if not path.startswith(FRONTEND_DIR) or not os.path.isfile(path):
            self.send_error(404)
            return
        with open(path, "rb") as fh:
            body = fh.read()
        ext = os.path.splitext(path)[1]
        self.send_response(200)
        self.send_header("Content-Type",
                         _CONTENT_TYPES.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = self.path.split("?", 1)[0]
        if route == "/api/scenes":
            self._send_json({"scenes": available_scenes()})
        elif route == "/api/devices":
            devices, gpu = discover_devices()
            self._send_json({"devices": devices, "gpu_status": gpu})
        elif route.startswith("/api/config/"):
            key = route.rsplit("/", 1)[-1]
            try:
                cfg, _ = reproducibility_config(key, 128, 128)
                self._send_json(cfg)
            except KeyError:
                self._send_json({"error": "unknown scene"}, 404)
        elif route in ("/", "/index.html"):
            self._send_file("index.html")
        elif route.startswith("/static/"):
            self._send_file(route[len("/static/"):])
        else:
            self.send_error(404)

    def do_POST(self):
        route = self.path.split("?", 1)[0]
        if route != "/api/render":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON"}, 400)
            return
        scene = payload.get("scene", "depth_near")
        devices = payload.get("devices") or ["cpu_ref", "vector_f32"]
        width = int(payload.get("width", 128))
        height = int(payload.get("height", width))
        trace = payload.get("trace_pixel")
        trace_pixel = tuple(trace) if trace else None
        try:
            result = render_job(scene, devices, width, height, trace_pixel)
        except KeyError:
            self._send_json({"error": "unknown scene: %s" % scene}, 404)
            return
        except ValueError as exc:
            self._send_json({"error": str(exc)}, 400)
            return
        self._send_json(result)

    def log_message(self, fmt, *args):
        # Keep the console readable; still record one-line access logs.
        print("[http] %s" % (fmt % args))


def main(host="127.0.0.1", port=8000):
    httpd = ThreadingHTTPServer((host, port), Handler)
    print("多端出图验证服务已启动: http://%s:%d" % (host, port))
    print("前端面板: 浏览器打开上述地址即可。Ctrl+C 停止。")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n停止服务。")
        httpd.shutdown()


if __name__ == "__main__":
    import sys
    h = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    p = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    main(h, p)
