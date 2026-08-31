# Why do the three renders disagree for gpt-oss on a tool-final append? (verl #7617)
# Prints the decoded text of each path so the difference is readable.
import json
import subprocess
import sys

if not __import__("os").path.isdir("/tmp/verl_main/verl"):
    subprocess.run("rm -rf /tmp/verl_main && git clone -q --depth 1 https://github.com/verl-project/verl /tmp/verl_main", shell=True, check=True)
sys.path.insert(0, "/tmp/verl_main")

from transformers import AutoTokenizer

from verl.utils.tokenizer.continuous_token_wiring import get_continuous_token_builder_class

MODEL = "openai/gpt-oss-20b"
FAMILY = "gptoss"
TOOLS = [{
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Look up a population figure.",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
    },
}]
TOOL_RESULT = "Pittsburgh had 302,971 residents at the latest census estimate."

previous = [
    {"role": "system", "content": "You are a helpful agent. Use tools to answer."},
    {"role": "user", "content": "Find the population of Pittsburgh."},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call-final", "type": "function", "function": {"name": "lookup", "arguments": {"q": "99"}}}]},
]
appended = [{"role": "tool", "content": TOOL_RESULT, "tool_call_id": "call-final", "name": "lookup"}]
updated = previous + appended


def folded_incremental(builder, previous, updated, tools):
    original_render = builder.render_delta_token_id
    original_delta = builder._tokenize_generation_prompt_delta

    def render_with_generation_prompt(prefix_messages, appended_messages, *, add_generation_prompt=False, tools=None):
        return original_render(prefix_messages, appended_messages, add_generation_prompt=True, tools=tools)

    builder.render_delta_token_id = render_with_generation_prompt
    builder._tokenize_generation_prompt_delta = lambda *args, **kwargs: []
    try:
        return builder.tokenize_non_assistant_incremental_messages(previous, updated, tools=tools)
    finally:
        builder.render_delta_token_id = original_render
        builder._tokenize_generation_prompt_delta = original_delta


tok = AutoTokenizer.from_pretrained(MODEL)
builder_cls = get_continuous_token_builder_class(FAMILY)
builder = builder_cls(tok)

truth = builder.render_delta_token_id(previous, appended, add_generation_prompt=True, tools=TOOLS)
current = builder.tokenize_non_assistant_incremental_messages(previous, updated, tools=TOOLS)
folded = folded_incremental(builder_cls(tok), previous, updated, TOOLS)
group_only = builder._tokenize_tool_group(appended, previous_messages=previous, tools=TOOLS)
gen_delta = builder._tokenize_generation_prompt_delta(updated, tools=TOOLS)

for name, ids in (
    ("truth  (full-history suffix diff)", truth),
    ("current(append group + full-history generation prompt)", current),
    ("folded (append group rendered with add_generation_prompt=True)", folded),
    ("   append group alone", group_only),
    ("   generation prompt alone", gen_delta),
):
    print(f"[TEXT] {name}: {tok.decode(ids)!r}")
print(f"[LEN] truth={len(truth)} current={len(current)} folded={len(folded)}")
print("[EQ] current==truth", current == truth, "folded==truth", folded == truth, "folded==current", folded == current)
print("DETAIL_DONE")
