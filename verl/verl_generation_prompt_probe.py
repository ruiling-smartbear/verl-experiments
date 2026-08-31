# Does the generation prompt need the full history? (verl #7617)
#
# For every text Continuous Token builder family, with the real tokenizer and
# chat template, compare three ways of producing the tokens appended by one
# incremental non-assistant append group:
#
#   truth   - render the whole conversation twice (prefix without / full with
#             add_generation_prompt) and take the suffix. Exact by definition.
#   current - what main does today: bounded render of the append group, plus a
#             separate full-history render for the generation prompt.
#   folded  - the proposal: render the append group once with
#             add_generation_prompt=True and use that suffix as-is. No
#             full-history render at all.
#
# It also prints the part of each chat template guarded by add_generation_prompt
# so the Jinja can be read next to the measurement. Tokenizers only, no weights.
import json
import os
import re
import subprocess
import sys
import tempfile

UPSTREAM_REPO = "https://github.com/verl-project/verl"
FAMILY_MODELS = [
    ("qwen", ["Qwen/Qwen2-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]),
    ("qwen25", ["Qwen/Qwen2.5-7B-Instruct"]),
    ("qwen3", ["Qwen/Qwen3-8B"]),
    ("qwen35", ["Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-35B-A3B"]),
    ("minimax", ["MiniMaxAI/MiniMax-Text-01"]),
    ("minimaxm2", ["MiniMaxAI/MiniMax-M2"]),
    ("glm47", ["zai-org/GLM-4.7", "zai-org/GLM-4.6"]),
    ("glm5", ["zai-org/GLM-5"]),
    ("gemma4", ["google/gemma-4-27b-it", "google/gemma-3-27b-it"]),
    ("gptoss", ["openai/gpt-oss-20b"]),
    ("deepseek", ["deepseek-ai/DeepSeek-V3.2-Exp", "deepseek-ai/DeepSeek-V3.1", "deepseek-ai/DeepSeek-V3"]),
    ("deepseekv4", ["deepseek-ai/DeepSeek-V4"]),
    ("default", ["Qwen/Qwen3-8B"]),
    ("default", ["mistralai/Mistral-7B-Instruct-v0.3", "microsoft/Phi-4-mini-instruct"]),
    ("default", ["HuggingFaceTB/SmolLM3-3B"]),
]

RUNNER = r'''
import json, re, sys
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


def history(prior_turns, arg_form, append_role):
    """sys + user + (assistant tool call + tool response) * n, then one append."""
    def arguments(index):
        payload = {"q": str(index)}
        return json.dumps(payload) if arg_form == "json" else payload

    messages = [
        {"role": "system", "content": "You are a helpful agent. Use tools to answer."},
        {"role": "user", "content": "Find the population of Pittsburgh."},
    ]
    for turn in range(prior_turns):
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"prior-{turn}", "type": "function",
             "function": {"name": "lookup", "arguments": arguments(turn)}}]})
        messages.append({"role": "tool", "content": TOOL_RESULT, "tool_call_id": f"prior-{turn}", "name": "lookup"})
    if append_role == "tool":
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": "call-final", "type": "function",
             "function": {"name": "lookup", "arguments": arguments(99)}}]})
        appended = [{"role": "tool", "content": TOOL_RESULT, "tool_call_id": "call-final", "name": "lookup"}]
    elif append_role == "user":
        messages.append({"role": "assistant", "content": "It has about 302,971 residents."})
        appended = [{"role": "user", "content": "And its ten largest suburbs?"}]
    else:
        messages.append({"role": "assistant", "content": "It has about 302,971 residents."})
        appended = [{"role": "system", "content": "Budget: three more tool calls."}]
    return messages, appended


def folded_incremental(builder, previous, updated, tools):
    """Encode the append group with add_generation_prompt=True, nothing else."""
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


def template_generation_prompt_snippet(tok):
    template = getattr(tok, "chat_template", None)
    if not isinstance(template, str):
        return None, None
    hits = [m.start() for m in re.finditer("add_generation_prompt", template)]
    if not hits:
        return None, None
    start = max(0, hits[-1] - 120)
    snippet = template[start : hits[-1] + 420]
    tail = template[hits[-1] :]
    history_dependent = bool(re.search(r"\b(loop\.|messages\[|ns\.|namespace\()", tail[:420]))
    return snippet, history_dependent


tok = AutoTokenizer.from_pretrained(model)
builder_cls = get_continuous_token_builder_class(family)
result = {"family": family, "model": model, "cases": [], "arg_form": None}
snippet, history_dependent = template_generation_prompt_snippet(tok)
result["template_snippet"] = snippet
result["template_looks_history_dependent"] = history_dependent

# Some templates want tool-call arguments as a mapping, others as a JSON string.
attempts = {}
for arg_form in ("dict", "json"):
    try:
        probe_builder = builder_cls(tok)
        messages, appended = history(0, arg_form, "tool")
        probe_builder.render_delta_token_id(messages, appended, add_generation_prompt=True, tools=TOOLS)
    except Exception as exc:  # noqa: BLE001
        attempts[arg_form] = f"{type(exc).__name__}: {str(exc)[:90]}"
        continue
    result["arg_form"] = arg_form
    break

if result["arg_form"] is None:
    # The full-history render is not prefix stable here; still measure the two
    # incremental paths against each other with whichever encoding renders the
    # append group at all.
    result["truth_error"] = attempts
    for arg_form in ("dict", "json"):
        try:
            probe_builder = builder_cls(tok)
            previous, appended = history(0, arg_form, "tool")
            probe_builder.tokenize_non_assistant_incremental_messages(previous, previous + appended, tools=TOOLS)
        except Exception:  # noqa: BLE001
            continue
        result["arg_form"] = arg_form
        break

if result["arg_form"] is None:
    result["error"] = f"no tool-call argument encoding renders with this template: {attempts}"
else:
    for tools in (None, TOOLS):
        for prior_turns in (0, 10):
            for append_role in ("tool", "user", "system"):
                case = {
                    "tools": bool(tools),
                    "prior_turns": prior_turns,
                    "append_role": append_role,
                }
                previous, appended = history(prior_turns, result["arg_form"], append_role)
                updated = previous + appended
                builder = builder_cls(tok)
                values = {}
                for name, thunk in (
                    ("truth", lambda: builder.render_delta_token_id(
                        previous, appended, add_generation_prompt=True, tools=tools)),
                    ("current", lambda: builder.tokenize_non_assistant_incremental_messages(
                        previous, updated, tools=tools)),
                    ("folded", lambda: folded_incremental(builder_cls(tok), previous, updated, tools)),
                ):
                    try:
                        values[name] = thunk()
                    except Exception as exc:  # noqa: BLE001
                        case[f"{name}_error"] = f"{type(exc).__name__}: {str(exc)[:90]}"
                if "truth" in values and "current" in values:
                    case["current_matches_truth"] = values["current"] == values["truth"]
                if "truth" in values and "folded" in values:
                    case["folded_matches_truth"] = values["folded"] == values["truth"]
                if "current" in values and "folded" in values:
                    case["folded_matches_current"] = values["folded"] == values["current"]
                try:
                    generation_prompt = builder._tokenize_generation_prompt_delta(updated, tools=tools)
                    case["generation_prompt_text"] = tok.decode(generation_prompt)
                    case["generation_prompt_len"] = len(generation_prompt)
                except Exception as exc:  # noqa: BLE001
                    case["generation_prompt_text"] = f"<{type(exc).__name__}>"
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
    print(f"[probe] upstream main @{head}")
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
            if data.get("error"):
                print(f"[skip] {family} {model}: {data['error']}")
                continue
            if data.get("truth_error"):
                print(f"[NOTE] {family} {model}: full-history render is not prefix stable here: {data['truth_error']}")
            print(
                f"[MODEL] {family} {model} arg_form={data['arg_form']} "
                f"template_looks_history_dependent={data['template_looks_history_dependent']}"
            )
            for case in data["cases"]:
                label = f"tools={case['tools']} prior_turns={case['prior_turns']} append={case['append_role']}"
                errors = {k[: -len("_error")]: v for k, v in case.items() if k.endswith("_error")}
                print(
                    f"[PROBE] {family:<10} {label:<48} "
                    f"current==truth={case.get('current_matches_truth')} "
                    f"folded==truth={case.get('folded_matches_truth')} "
                    f"folded==current={case.get('folded_matches_current')} "
                    f"gen_prompt={case.get('generation_prompt_text')!r}"
                    + (f" errors={errors}" if errors else "")
                )
            if data.get("template_snippet"):
                snippet = " ".join(data["template_snippet"].split())
                print(f"[JINJA] {family} {model}: {snippet[:400]}")
            break
    print("PROBE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
