# verl #7617: can the generation prompt come out of the last append group's render?
#
# Today, tokenize_non_assistant_incremental_messages renders each append group
# against a bounded synthetic prefix, then renders the FULL history twice
# (add_generation_prompt False / True) to obtain the generation prompt.
#
# This compares that against one change: keep everything as is, but render the
# LAST append group with add_generation_prompt=True and drop the two full-history
# renders. Same tokenizer, same messages, upstream main, token ids compared.
# Tokenizers only, no weights, CPU.
import json
import os
import subprocess
import sys
import tempfile

UPSTREAM_REPO = "https://github.com/verl-project/verl"
FAMILY_MODELS = [
    ("qwen", ["Qwen/Qwen2-7B-Instruct"]),
    ("qwen25", ["Qwen/Qwen2.5-7B-Instruct"]),
    ("qwen3", ["Qwen/Qwen3-8B"]),
    ("qwen35", ["Qwen/Qwen3.5-9B"]),
    ("minimaxm2", ["MiniMaxAI/MiniMax-M2"]),
    ("glm47", ["zai-org/GLM-4.7"]),
    ("gptoss", ["openai/gpt-oss-20b"]),
    ("deepseek", ["deepseek-ai/DeepSeek-V3.2-Exp", "deepseek-ai/DeepSeek-V3"]),
    ("default", ["HuggingFaceTB/SmolLM3-3B"]),
    ("default", ["Qwen/Qwen3-8B"]),
]

RUNNER = r'''
import json, sys
verl_path, family, model, out = sys.argv[1:5]
sys.path.insert(0, verl_path)
from transformers import AutoTokenizer
from verl.utils.tokenizer.continuous_token_wiring import get_continuous_token_builder_class

TOOLS = [{
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Look up a population figure.",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
    },
}]
TOOL_RESULT = "Pittsburgh had 302,971 residents at the latest census estimate."
# What gets appended incrementally, as a list of append groups.
APPENDS = {
    "tool": [[{"role": "tool", "content": TOOL_RESULT, "tool_call_id": "call-a", "name": "lookup"}]],
    "tool,tool": [[
        {"role": "tool", "content": TOOL_RESULT, "tool_call_id": "call-a", "name": "lookup"},
        {"role": "tool", "content": "Second lookup: 1,244,000 in the metro area.", "tool_call_id": "call-b", "name": "lookup"},
    ]],
    "user": [[{"role": "user", "content": "And its ten largest suburbs?"}]],
    "system": [[{"role": "system", "content": "Budget: three more tool calls."}]],
    "tool+user": [
        [{"role": "tool", "content": TOOL_RESULT, "tool_call_id": "call-a", "name": "lookup"}],
        [{"role": "user", "content": "Thanks, keep going."}],
    ],
}


def arguments(index, arg_form):
    payload = {"q": str(index)}
    return json.dumps(payload) if arg_form == "json" else payload


def previous_messages(prior_turns, arg_form, append_kind):
    messages = [
        {"role": "system", "content": "You are a helpful agent. Use tools to answer."},
        {"role": "user", "content": "Find the population of Pittsburgh."},
    ]
    for turn in range(prior_turns):
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"prior-{turn}", "type": "function",
             "function": {"name": "lookup", "arguments": arguments(turn, arg_form)}}]})
        messages.append({"role": "tool", "content": TOOL_RESULT, "tool_call_id": f"prior-{turn}", "name": "lookup"})
    if append_kind.startswith("tool"):
        calls = [{"id": "call-a", "type": "function", "function": {"name": "lookup", "arguments": arguments(98, arg_form)}}]
        if append_kind == "tool,tool":
            calls.append({"id": "call-b", "type": "function", "function": {"name": "lookup", "arguments": arguments(99, arg_form)}})
        messages.append({"role": "assistant", "content": "", "tool_calls": calls})
    else:
        messages.append({"role": "assistant", "content": "It has about 302,971 residents."})
    return messages


def folded_incremental(builder, previous, updated, tools, group_count):
    """Render the last append group with add_generation_prompt=True; skip the full-history renders."""
    original_render = builder.render_delta_token_id
    original_delta = builder._tokenize_generation_prompt_delta
    state = {"calls": 0}

    def render(prefix_messages, appended_messages, *, add_generation_prompt=False, tools=None):
        state["calls"] += 1
        last_group = state["calls"] == group_count
        return original_render(prefix_messages, appended_messages, add_generation_prompt=last_group, tools=tools)

    builder.render_delta_token_id = render
    builder._tokenize_generation_prompt_delta = lambda *args, **kwargs: []
    try:
        ids = builder.tokenize_non_assistant_incremental_messages(previous, updated, tools=tools)
    finally:
        builder.render_delta_token_id = original_render
        builder._tokenize_generation_prompt_delta = original_delta
    return ids, state["calls"]


tok = AutoTokenizer.from_pretrained(model)
builder_cls = get_continuous_token_builder_class(family)
result = {"family": family, "model": model, "cases": []}

# Templates disagree on whether tool-call arguments are a mapping or a JSON string.
arg_form = None
for candidate in ("dict", "json"):
    try:
        previous = previous_messages(1, candidate, "tool")
        builder_cls(tok).tokenize_non_assistant_incremental_messages(previous, previous + APPENDS["tool"][0], tools=TOOLS)
        arg_form = candidate
        break
    except Exception as exc:  # noqa: BLE001
        result.setdefault("arg_form_errors", {})[candidate] = f"{type(exc).__name__}: {str(exc)[:90]}"
result["arg_form"] = arg_form

kwargs_variants = [{}]
if "enable_thinking" in (getattr(tok, "chat_template", "") or ""):
    kwargs_variants.append({"enable_thinking": False})

for kwargs in kwargs_variants:
    for tools in (None, TOOLS):
        for prior_turns in (0, 10):
            for append_kind, groups in APPENDS.items():
                case = {"kwargs": kwargs, "tools": bool(tools), "prior_turns": prior_turns, "append": append_kind}
                previous = previous_messages(prior_turns, arg_form or "dict", append_kind)
                appended = [message for group in groups for message in group]
                updated = previous + appended
                try:
                    current = builder_cls(tok, chat_template_kwargs=dict(kwargs)).tokenize_non_assistant_incremental_messages(
                        previous, updated, tools=tools)
                except Exception as exc:  # noqa: BLE001
                    case["current_error"] = f"{type(exc).__name__}: {str(exc)[:90]}"
                try:
                    folded, calls = folded_incremental(
                        builder_cls(tok, chat_template_kwargs=dict(kwargs)), previous, updated, tools, len(groups))
                    case["template_renders"] = calls
                except Exception as exc:  # noqa: BLE001
                    case["folded_error"] = f"{type(exc).__name__}: {str(exc)[:90]}"
                if "current_error" not in case and "folded_error" not in case:
                    case["same"] = folded == current
                    if not case["same"]:
                        case["current_text"] = tok.decode(current)
                        case["folded_text"] = tok.decode(folded)
                    else:
                        case["text"] = tok.decode(current)
                result["cases"].append(case)

with open(out, "w") as fh:
    json.dump(result, fh)
'''


def clone(repo: str, path: str) -> str:
    if not os.path.isdir(os.path.join(path, "verl")):
        subprocess.run(f"rm -rf {path} && git clone -q --depth 1 {repo} {path}", shell=True, check=True)
    return subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"], text=True).strip()


def main() -> int:
    head = clone(UPSTREAM_REPO, "/tmp/verl_main")
    print(f"[fold] upstream main @{head}")
    for family, candidates in FAMILY_MODELS:
        for model in candidates:
            out = tempfile.mktemp(suffix=".json")
            proc = subprocess.run(
                [sys.executable, "-c", RUNNER, "/tmp/verl_main", family, model, out],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                tail = ((proc.stderr or "").strip().splitlines() or [""])[-1]
                print(f"[skip] {family} {model}: {tail[:120]}")
                continue
            with open(out) as fh:
                data = json.load(fh)
            cases = data["cases"]
            same = sum(1 for c in cases if c.get("same") is True)
            differ = [c for c in cases if c.get("same") is False]
            errors = [c for c in cases if "same" not in c]
            texts = sorted({c.get("text", "") for c in cases if c.get("same")})
            print(
                f"[FOLD] {family:<10} {model:<30} arg_form={data['arg_form']} "
                f"same={same}/{len(cases)} differ={len(differ)} not_rendered={len(errors)}"
            )
            for c in differ:
                print(f"[DIFF] {family} kwargs={c['kwargs']} tools={c['tools']} prior={c['prior_turns']} append={c['append']}")
                print(f"[DIFF]    current: {c['current_text']!r}")
                print(f"[DIFF]    folded : {c['folded_text']!r}")
            seen = set()
            for c in errors:
                key = (c.get("current_error"), c.get("folded_error"))
                if key in seen:
                    continue
                seen.add(key)
                which = "both" if c.get("current_error") and c.get("folded_error") else ("current" if c.get("current_error") else "folded")
                print(f"[ERR] {family} append={c['append']} tools={c['tools']} {which}: {c.get('current_error') or c.get('folded_error')}")
            for text in texts[:3]:
                print(f"[TEXT] {family}: {text[-90:]!r}")
            break
    print("FOLD_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
