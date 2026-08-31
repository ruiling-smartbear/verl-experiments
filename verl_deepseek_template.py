# Where exactly does DeepSeek-V3.2-Exp's template fail on verl's synthetic tool prefix?
import os
import subprocess
import sys
import traceback

if not os.path.isdir("/tmp/verl_main/verl"):
    subprocess.run("rm -rf /tmp/verl_main && git clone -q --depth 1 https://github.com/verl-project/verl /tmp/verl_main", shell=True, check=True)
sys.path.insert(0, "/tmp/verl_main")

import jinja2
from transformers import AutoTokenizer

from verl.utils.tokenizer.continuous_token import _SYNTHETIC_SYSTEM_MESSAGE, _SYNTHETIC_USER_MESSAGE
from verl.utils.tokenizer.continuous_token_wiring import get_continuous_token_builder_class

MODEL = "deepseek-ai/DeepSeek-V3.2-Exp"
tok = AutoTokenizer.from_pretrained(MODEL)
template = tok.chat_template
builder = get_continuous_token_builder_class("deepseek")(tok)

tool = {"role": "tool", "content": "Pittsburgh had 302,971 residents.", "tool_call_id": "call-0-0"}
synthetic_assistant = builder._synthetic_assistant_for_tools([tool])
print("[synthetic assistant]", synthetic_assistant)

# Print every template line that touches tool calls / tool messages / arguments.
for number, line in enumerate(template.splitlines(), 1):
    if any(key in line for key in ("tool_calls", "arguments", "role == 'tool'", 'role == "tool"', "tool▁", "reasoning_content")):
        print(f"[TPL {number}] {' '.join(line.split())[:220]}")

env = jinja2.Environment()
env.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(jinja2.TemplateError(message))
compiled = env.from_string(template)
for label, messages in (
    ("synthetic prefix only", [_SYNTHETIC_SYSTEM_MESSAGE, _SYNTHETIC_USER_MESSAGE, synthetic_assistant]),
    ("synthetic prefix + tool", [_SYNTHETIC_SYSTEM_MESSAGE, _SYNTHETIC_USER_MESSAGE, synthetic_assistant, tool]),
    ("harness-shaped assistant + tool", [
        {"role": "system", "content": "You are a helpful agent."},
        {"role": "user", "content": "Find the population of Pittsburgh."},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": {"q": "0"}}, "id": "call-0-0"}]},
        tool,
    ]),
):
    try:
        out = compiled.render(messages=messages, add_generation_prompt=False, bos_token=tok.bos_token or "")
        print(f"[RENDER ok] {label}: ...{out[-160:]!r}")
    except Exception as exc:  # noqa: BLE001
        tb = traceback.extract_tb(exc.__traceback__)
        template_frames = [frame for frame in tb if frame.filename == "<template>"]
        where = f"template line {template_frames[-1].lineno}" if template_frames else "no template frame"
        print(f"[RENDER FAIL] {label}: {type(exc).__name__}: {str(exc)[:100]} ({where})")
        if template_frames:
            lineno = template_frames[-1].lineno
            lines = template.splitlines()
            for number in range(max(1, lineno - 1), min(len(lines), lineno + 1) + 1):
                print(f"[TPL {number}] {' '.join(lines[number - 1].split())[:220]}")
print("DEEPSEEK_TEMPLATE_DONE")
