"""Port harbor 290ab985's Modal environment off Modal's retired legacy Sandbox filesystem API.

Modal >= 1.6 removed Sandbox.mkdir / Sandbox.ls (the server already rejects them with
"The legacy Sandbox filesystem API is no longer supported"). The pinned harbor calls them in four
places; this maps them onto Sandbox.filesystem (make_directory / stat). Everything else in the
pinned harbor, including the terminus-2 agent, is unchanged.

Usage: python patch_harbor_modal.py <harbor>/src/harbor/environments/modal.py   (idempotent)
"""
import sys
from pathlib import Path

path = Path(sys.argv[1])
src = path.read_text()
MARK = "# [graft-generalization] modal new-fs-api"
if MARK in src:
    print("already patched"); sys.exit(0)

edits = [
    ('''        try:
            await self._env._sandbox.ls.aio(path)
            return True
        except (NotADirectoryError, FileNotFoundError):
            return False
''', f'''        {MARK}
        try:
            info = await self._env._sandbox.filesystem.stat.aio(path)
        except Exception:
            return False
        return info.type.value == "directory"
'''),
    ('''        try:
            await self._env._sandbox.ls.aio(path)
            return False
        except NotADirectoryError:
            return True
        except FileNotFoundError:
            return False
''', f'''        {MARK}
        try:
            info = await self._env._sandbox.filesystem.stat.aio(path)
        except Exception:
            return False
        return info.type.value == "file"
'''),
    ('''        await env._sandbox.mkdir.aio(str(EnvironmentPaths.agent_dir), parents=True)
        await env._sandbox.mkdir.aio(str(EnvironmentPaths.verifier_dir), parents=True)
''', f'''        {MARK}
        await env._sandbox.filesystem.make_directory.aio(str(EnvironmentPaths.agent_dir), create_parents=True)
        await env._sandbox.filesystem.make_directory.aio(str(EnvironmentPaths.verifier_dir), create_parents=True)
'''),
]
for old, new in edits:
    assert src.count(old) == 1, f"anchor not found exactly once: {old[:70]!r}"
    src = src.replace(old, new)
assert "_sandbox.ls." not in src and "_sandbox.mkdir." not in src, "legacy calls remain"
path.write_text(src)
print("patched", path)
