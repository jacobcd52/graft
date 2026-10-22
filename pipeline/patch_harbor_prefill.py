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
  GRAFT_PREFILL=french    prefix '{\\n  "analysis": "' + one forced token: the most likely token (top-20
                          logprobs at that position, one extra max_tokens=1 request) that is in a
                          whitelist of unambiguous French sentence openers, else "Je"; the rest of
                          the reply is sampled freely (narration-language probe for the FREN arms)

Each call appends {"mode", "tries", "empty_final"} (french: {"mode", "token", "rank", "logprob"}; rank
None = no French opener in the top 20, fell back to "Je") to GRAFT_PREFILL_LOG if set.
Unset GRAFT_PREFILL leaves harbor's behavior unchanged.

Usage: python patch_harbor_prefill.py <harbor>/src/harbor/llms/lite_llm.py   (idempotent)
"""
import sys
from pathlib import Path

path = Path(sys.argv[1])
src = path.read_text()
MARK = "# [graft-generalization] prefill"
FR_MARK = "# [graft-generalization] prefill-french"
if FR_MARK in src:
    print("already patched"); sys.exit(0)
base_done = MARK in src

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
for old, new in ([] if base_done else edits):
    if src.count(old) != 1:
        sys.exit(f"patch anchor not found exactly once: {old[:60]!r}")
    src = src.replace(old, new)

if not base_done:
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

# French mode: wraps the base prefill call (call sites resolve _graft_prefill_call at call time).
src += f'''

{FR_MARK}
_GRAFT_PREFIX["french"] = _GRAFT_PREFIX["reasoned"]
# Unambiguous French sentence openers that are single Qwen3 tokens right after '"analysis": "'.
_GRAFT_FR_OPENERS = set("""Je Nous La Le Les Il Ils Cette Cet Ces Dans Une Des Du Toutes Tous Avec Sur
    État Bien""".split())


async def _graft_french_token(llm, prompt, message_history, prefix):
    import httpx
    body = {{"model": llm._model_name.split("/", 1)[-1], "max_tokens": 1, "temperature": 0,
            "logprobs": True, "top_logprobs": 20, "continue_final_message": True,
            "add_generation_prompt": False, "chat_template_kwargs": {{"enable_thinking": False}},
            "messages": list(message_history) + [{{"role": "user", "content": prompt}},
                                                 {{"role": "assistant", "content": prefix}}]}}
    async with httpx.AsyncClient(timeout=600) as c:
        r = await c.post((llm._api_base or "http://localhost:8000/v1").rstrip("/") + "/chat/completions", json=body)
        r.raise_for_status()
    tops = r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    for rank, t in enumerate(sorted(tops, key=lambda t: -t["logprob"])):
        if t["token"].strip() in _GRAFT_FR_OPENERS and not t["token"].startswith(" "):
            return {{"token": t["token"], "rank": rank, "logprob": t["logprob"]}}
    return {{"token": "Je", "rank": None, "logprob": None}}


_graft_prefill_call_plain = _graft_prefill_call


async def _graft_prefill_call(llm, mode, prompt, message_history, response_format, logging_path, **kwargs):
    if mode != "french":
        return await _graft_prefill_call_plain(llm, mode, prompt, message_history, response_format,
                                               logging_path, **kwargs)
    pick = await _graft_french_token(llm, prompt, message_history, _GRAFT_PREFIX["french"])
    prefix = _GRAFT_PREFIX["french"] + pick["token"]
    extra_body = {{**kwargs.pop("extra_body", {{}}), "continue_final_message": True, "add_generation_prompt": False}}
    resp = await llm.call(prompt, message_history, response_format, logging_path, _graft_inner=True,
                          _graft_prefix_msgs=[{{"role": "assistant", "content": prefix}}],
                          extra_body=extra_body, **kwargs)
    resp.content = prefix + resp.content
    if os.environ.get("GRAFT_PREFILL_LOG"):
        with open(os.environ["GRAFT_PREFILL_LOG"], "a") as f:
            f.write(json.dumps({{"mode": mode, **pick}}) + "\\n")
    return resp
'''
path.write_text(src)
print("patched", path)
