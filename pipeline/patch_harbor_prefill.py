"""Add response prefill to harbor's LiteLLM client, for the commands-only (format) arms.

With GRAFT_PREFILL set in the harbor process's environment, every LLM call continues a partial
assistant reply instead of starting a fresh one (vLLM: continue_final_message), and the prefix is
glued back on before terminus-2 parses the reply, so the transcript looks like ordinary output:

  GRAFT_PREFILL=commands  prefix '{\\n  "analysis": "",\\n  "plan": "",\\n  "commands": ['
                          -> reasoning fields forced empty (the stripped training format)
  GRAFT_PREFILL=reasoned  prefix '{\\n  "analysis": "'
                          -> if the model closes the analysis immediately (empty), the turn is
                             re-sampled, up to GRAFT_PREFILL_RETRIES (default 4) extra tries

The prefixes match pipeline/strip_reasoning.py / data_prep.py's json.dumps(indent=2) layout, and
Qwen3's chat template renders "prompt + prefix" token-identically to the training transcripts.
Each call appends {"mode", "tries", "empty_final"} to GRAFT_PREFILL_LOG if set.
Unset GRAFT_PREFILL leaves harbor's behavior unchanged.

Usage: python patch_harbor_prefill.py <harbor>/src/harbor/llms/lite_llm.py   (idempotent)
"""
import sys
from pathlib import Path

path = Path(sys.argv[1])
src = path.read_text()
MARK = "# [graft-generalization] prefill"
if MARK in src:
    print("already patched"); sys.exit(0)

edits = [
    ("import hashlib\nimport json\n", f"import hashlib\nimport json\nimport os  {MARK}\n"),
    ('''        **kwargs,
    ) -> LLMResponse:
        if self._use_responses_api:''', f'''        **kwargs,
    ) -> LLMResponse:
        {MARK}
        mode = os.environ.get("GRAFT_PREFILL", "")
        if mode and not kwargs.pop("_graft_inner", False):
            return await _graft_prefill_call(self, mode, prompt, message_history, response_format,
                                             logging_path, **kwargs)
        prefix_msgs = kwargs.pop("_graft_prefix_msgs", [])
        if self._use_responses_api:'''),
    ('''        messages = message_history + [{"role": "user", "content": prompt}]
''', '''        messages = message_history + [{"role": "user", "content": prompt}] + prefix_msgs
'''),
]
for old, new in edits:
    if src.count(old) != 1:
        sys.exit(f"patch anchor not found exactly once: {old[:60]!r}")
    src = src.replace(old, new)

src += f'''

{MARK}
_GRAFT_PREFIX = {{
    "commands": '{{\\n  "analysis": "",\\n  "plan": "",\\n  "commands": [',
    "reasoned": '{{\\n  "analysis": "',
}}


async def _graft_prefill_call(llm, mode, prompt, message_history, response_format, logging_path, **kwargs):
    prefix = _GRAFT_PREFIX[mode]
    extra_body = {{**kwargs.pop("extra_body", {{}}), "continue_final_message": True, "add_generation_prompt": False}}
    retries = int(os.environ.get("GRAFT_PREFILL_RETRIES", 4)) if mode == "reasoned" else 0
    for tries in range(1, retries + 2):
        resp = await llm.call(prompt, message_history, response_format, logging_path, _graft_inner=True,
                              _graft_prefix_msgs=[{{"role": "assistant", "content": prefix}}],
                              extra_body=dict(extra_body), **kwargs)
        empty = mode == "reasoned" and resp.content.lstrip().startswith('"')
        if not empty:
            break
    resp.content = prefix + resp.content
    if os.environ.get("GRAFT_PREFILL_LOG"):
        with open(os.environ["GRAFT_PREFILL_LOG"], "a") as f:
            f.write(json.dumps({{"mode": mode, "tries": tries, "empty_final": empty}}) + "\\n")
    return resp
'''
path.write_text(src)
print("patched", path)
