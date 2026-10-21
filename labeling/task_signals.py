"""Second, non-LLM signals per task, used to make arm splits strict (labeling/strict.py).

From each Terminal Wrench task's original_task/ (first source model's copy):
  py_solution     the reference solution (solution/solve.sh) writes or runs Python
  other_solution  it writes or builds another language (C/C++/Rust/Go/JS/TS/Java/Ruby/SQL/R)
  security_terms  number of distinct security terms in instruction.md
  security_adjacent_terms  distinct security-adjacent operations in the reference solution, tests and
                  Dockerfile (ownership, sudo, users, ssh, keys, firewall, explicit permission modes, ...);
                  plain `chmod +x` and ca-certificates do not count

Usage: python task_signals.py <terminal-wrench> data/labels/task_signals.json
"""
import glob, json, os, re, sys

SEC = re.compile(r"\b(security|secure|vulnerab\w*|exploit\w*|attack\w*|malicious|password\w*|credential\w*|privilege\w*|"
                 r"sudoers|firewall|iptables|ssl|tls|certificate\w*|encrypt\w*|cve|authenticat\w*|permission\w*|"
                 r"hardening|audit\w*|selinux|apparmor|ssh)\b", re.I)
ADJ = re.compile(r"\b(chown|chgrp|setuid|setgid|sudo|sudoers|useradd|adduser|usermod|groupadd|passwd|/etc/shadow|ssh\w*|"
                 r"authorized_keys|openssl|gpg|\.pem|iptables|ufw|nftables|firewall|selinux|apparmor|umask|setfacl|"
                 r"capabilit(y|ies)|privilege\w*|secret\w*|api[_ ]key|htpasswd|authenticat\w*|login|pam)\b|"
                 r"chmod\s+[0-7]*[4267][0-7]{2}\b|chmod\s+[ugoa]*[+-]s", re.I)
PY = re.compile(r"\.py\b|python3?\s+(-c|<<|-)")
OTHER = re.compile(r"\.(c|cc|cpp|h|rs|go|js|ts|java|rb|sql|R)\b['\"\s]|gcc |g\+\+ |cargo |go build|node |rustc ")


def read_first(tw, task, rel):
    f = sorted(glob.glob(f"{tw}/tasks/{task}/*/original_task/{rel}"))
    return open(f[0], errors="ignore").read() if f else ""


def read_all(tw, task, pats):  # every matching file in the first source model's copy of the task
    d = sorted(glob.glob(f"{tw}/tasks/{task}/*/original_task"))
    out = ""
    for p in pats if d else []:
        for f in glob.glob(f"{d[0]}/{p}"):
            if os.path.isfile(f) and os.path.getsize(f) < 2e6:
                out += open(f, errors="ignore").read()
    return out


def main():
    tw, out = sys.argv[1], sys.argv[2]
    res = {}
    for d in sorted(glob.glob(f"{tw}/tasks/*")):
        t = d.rsplit("/", 1)[1]
        sol, ins = read_first(tw, t, "solution/solve.sh"), read_first(tw, t, "instruction.md")
        res[t] = {"py_solution": bool(PY.search(sol)), "other_solution": bool(OTHER.search(sol)),
                  "security_terms": len({m.lower() for m in SEC.findall(ins)}),
                  "security_adjacent_terms": len({m.group(0).lower() for m in ADJ.finditer(
                      read_all(tw, t, ["solution/solve.sh", "tests/*", "environment/Dockerfile"]))})}
    json.dump(res, open(out, "w"), indent=0)
    print(f"{len(res)} tasks -> {out}")


if __name__ == "__main__":
    main()
