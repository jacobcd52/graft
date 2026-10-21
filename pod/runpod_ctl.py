"""Local controller for the RunPod pod running pod/agent.py.

  python runpod_ctl.py create --name NAME --gpu "NVIDIA H200" --count 8 [--disk 600]
  python runpod_ctl.py status|stop|terminate POD_ID
  python runpod_ctl.py exec POD_ID 'cmd'            (short; < ~80 s)
  python runpod_ctl.py spawn POD_ID NAME 'cmd'      (background; log in /workspace/logs/NAME.log)
  python runpod_ctl.py log POD_ID NAME [--tail 50]
  python runpod_ctl.py put POD_ID LOCAL REMOTE
  python runpod_ctl.py get POD_ID REMOTE LOCAL

Credentials: RUNPOD_API_KEY (or --key-file), and the per-pod agent token, stored with the pod id
in pod/.pods.json (gitignored).
"""
import argparse, base64, fcntl, json, os, secrets, sys, time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
STATE = HERE / ".pods.json"
REST = "https://rest.runpod.io/v1"
# CUDA 12.4 image: the 12.8 image ships forward-compat libcuda that breaks on hosts with driver < 570
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def api_key(a):
    return os.environ.get("RUNPOD_API_KEY") or Path(a.key_file).read_text().strip()


def state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def agent(pod_id):
    tok = state()[pod_id]["token"]
    return f"https://{pod_id}-8888.proxy.runpod.net", {"Authorization": f"Bearer {tok}"}


def create(a):
    token = secrets.token_urlsafe(32)
    code = base64.b64encode((HERE / "agent.py").read_bytes()).decode()
    body = {
        "name": a.name, "imageName": IMAGE, "gpuTypeIds": a.gpu.split(","), "gpuTypePriority": "custom", "gpuCount": a.count,
        "cloudType": a.cloud, "containerDiskInGb": a.local_disk, "volumeInGb": a.disk, "volumeMountPath": "/workspace",
        "ports": ["8888/http"],
        **({"allowedCudaVersions": a.cuda.split(",")} if a.cuda else {}),
        "env": {"AGENT_TOKEN": token, "AGENT_B64": code, "IDLE_MIN": str(a.idle_min), "HANG_MIN": str(a.hang_min)},
        "dockerStartCmd": ["bash", "-c", "echo $AGENT_B64 | base64 -d > /agent.py && exec python3 /agent.py"],
    }
    r = requests.post(f"{REST}/pods", json=body, headers={"Authorization": f"Bearer {api_key(a)}"}, timeout=60)
    if r.status_code >= 300:
        sys.exit(f"create failed {r.status_code}: {r.text[:500]}")
    pod = r.json()
    with open(HERE / ".pods.lock", "w") as lk:  # parallel launches share the state file
        fcntl.flock(lk, fcntl.LOCK_EX)
        s = state(); s[pod["id"]] = {"token": token, "name": a.name}; STATE.write_text(json.dumps(s, indent=1))
    STATE.chmod(0o600)
    print("pod", pod["id"], "costPerHr", pod.get("costPerHr"))
    url, hdr = agent(pod["id"])
    for _ in range(90):
        try:
            if requests.get(f"{url}/health", headers=hdr, timeout=10).status_code == 200:
                print("agent ready:", url); return
        except requests.RequestException:
            pass
        time.sleep(10)
    if a.keep_unready:
        print("agent not reachable yet; check status later"); return
    # Some hosts never finish starting (secure pods stuck pulling, broken CUDA) yet bill from creation.
    print("agent not reachable after 15 min: terminating pod", pod["id"])
    requests.delete(f"{REST}/pods/{pod['id']}", headers={"Authorization": f"Bearer {api_key(a)}"}, timeout=60)
    sys.exit(1)


def pod_req(a, method, suffix=""):
    r = requests.request(method, f"{REST}/pods/{a.pod}{suffix}", headers={"Authorization": f"Bearer {api_key(a)}"}, timeout=60)
    print(r.status_code, r.text[:1500])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key-file", default=os.environ.get("RUNPOD_KEY_FILE", ""))
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create"); c.add_argument("--name", required=True); c.add_argument("--gpu", required=True)
    c.add_argument("--count", type=int, default=1); c.add_argument("--disk", type=int, default=100)
    c.add_argument("--local-disk", type=int, default=50, help="container (local) disk GB: code, venvs, model cache")
    c.add_argument("--cloud", default="COMMUNITY")
    # RunPod's REST allowedCudaVersions filter matched no machines at all when tested; off by default
    c.add_argument("--cuda", default="", help="allowed host CUDA versions, comma-separated ('' = any)")
    c.add_argument("--idle-min", type=float, default=30, help="pod stops itself after this long idle (0 = never)")
    c.add_argument("--hang-min", type=float, default=90, help="pod stops itself if pipeline logs are silent this long")
    c.add_argument("--keep-unready", action="store_true", help="do not terminate a pod whose agent never comes up")
    for n in ("status", "stop", "start", "terminate"):
        sub.add_parser(n).add_argument("pod")
    e = sub.add_parser("exec"); e.add_argument("pod"); e.add_argument("command"); e.add_argument("--timeout", type=int, default=80)
    s = sub.add_parser("spawn"); s.add_argument("pod"); s.add_argument("name"); s.add_argument("command")
    l = sub.add_parser("log"); l.add_argument("pod"); l.add_argument("name"); l.add_argument("--tail", type=int, default=40)
    p = sub.add_parser("put"); p.add_argument("pod"); p.add_argument("local"); p.add_argument("remote")
    g = sub.add_parser("get"); g.add_argument("pod"); g.add_argument("remote"); g.add_argument("local")
    a = ap.parse_args()

    if a.cmd == "create":
        return create(a)
    if a.cmd == "status":
        return pod_req(a, "GET")
    if a.cmd == "stop":
        return pod_req(a, "POST", "/stop")
    if a.cmd == "start":
        return pod_req(a, "POST", "/start")
    if a.cmd == "terminate":
        return pod_req(a, "DELETE")
    url, hdr = agent(a.pod)
    if a.cmd == "exec":
        r = requests.post(f"{url}/exec", json={"cmd": a.command, "timeout": a.timeout}, headers=hdr, timeout=a.timeout + 20)
        j = r.json(); print(j.get("out", j), end=""); sys.exit(0 if j.get("rc") == 0 else 1)
    if a.cmd == "spawn":
        print(requests.post(f"{url}/spawn", json={"cmd": a.command, "name": a.name}, headers=hdr, timeout=60).json())
    if a.cmd == "log":
        r = requests.post(f"{url}/exec", json={"cmd": f"tail -n {a.tail} /workspace/logs/{a.name}.log"}, headers=hdr, timeout=90)
        print(r.json().get("out", ""), end="")
    if a.cmd == "put":
        r = requests.put(f"{url}/file", params={"path": a.remote}, data=Path(a.local).read_bytes(), headers=hdr, timeout=600)
        print(r.json())
    if a.cmd == "get":
        r = requests.get(f"{url}/file", params={"path": a.remote}, headers=hdr, timeout=600)
        if r.status_code != 200:
            sys.exit(r.text[:300])
        Path(a.local).write_bytes(r.content); print(f"wrote {a.local} ({len(r.content)} bytes)")


if __name__ == "__main__":
    main()
