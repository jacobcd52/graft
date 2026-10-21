"""Minimal command server for a RunPod pod, reached through RunPod's HTTPS proxy
(https://<pod-id>-8888.proxy.runpod.net). Every request needs the header
`Authorization: Bearer $AGENT_TOKEN`. Standard library only.

  POST /exec   {"cmd": str, "timeout": int}  -> {"rc", "out"}   (short commands; proxy times out ~100 s)
  POST /spawn  {"cmd": str, "name": str}     -> {"pid", "log"}  (long jobs; output to /workspace/logs/<name>.log)
  GET  /file?path=...                        -> raw bytes
  PUT  /file?path=...                        <- raw bytes (parent dirs created)
"""
import hmac, json, os, subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

TOKEN = os.environ["AGENT_TOKEN"]
LOGS = Path("/workspace/logs")
LOGS.mkdir(parents=True, exist_ok=True)


class H(BaseHTTPRequestHandler):
    def _auth(self):
        got = self.headers.get("Authorization", "")
        if not hmac.compare_digest(got, f"Bearer {TOKEN}"):
            self._send(401, {"error": "unauthorized"})
            return False
        return True

    def _send(self, code, obj=None, raw=None):
        body = raw if raw is not None else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/octet-stream" if raw is not None else "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def do_GET(self):
        if not self._auth():
            return
        u = urlparse(self.path)
        if u.path == "/health":
            return self._send(200, {"ok": True})
        if u.path == "/file":
            p = Path(parse_qs(u.query)["path"][0])
            if not p.is_file():
                return self._send(404, {"error": f"no file {p}"})
            return self._send(200, raw=p.read_bytes())
        self._send(404, {"error": "unknown path"})

    def do_PUT(self):
        if not self._auth():
            return
        u = urlparse(self.path)
        p = Path(parse_qs(u.query)["path"][0])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self._body())
        self._send(200, {"written": str(p), "bytes": p.stat().st_size})

    def do_POST(self):
        if not self._auth():
            return
        req = json.loads(self._body() or b"{}")
        u = urlparse(self.path)
        if u.path == "/exec":
            try:
                r = subprocess.run(["bash", "-lc", req["cmd"]], capture_output=True, text=True,
                                   timeout=req.get("timeout", 80))
                return self._send(200, {"rc": r.returncode, "out": (r.stdout + r.stderr)[-200000:]})
            except subprocess.TimeoutExpired as e:
                return self._send(200, {"rc": None, "out": f"timeout; partial: {(e.stdout or b'')[-5000:]!r}"})
        if u.path == "/spawn":
            log = LOGS / f"{req['name']}.log"
            with open(log, "ab") as f:
                p = subprocess.Popen(["bash", "-lc", req["cmd"]], stdout=f, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            return self._send(200, {"pid": p.pid, "log": str(log)})
        self._send(404, {"error": "unknown path"})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("AGENT_PORT", 8888))), H).serve_forever()
