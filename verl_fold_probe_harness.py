# verl #7617: can the generation prompt come out of the last append group's render?
# Harness-shaped edition: every trajectory is built the way
# verl/experimental/agent_loop/tool_agent_loop.py builds it - an optional system
# prompt, a user prompt, then per turn an assistant message whose tool_calls carry
# a JSON-object `arguments` (OpenAIFunctionCallSchema.model_dump()) and one tool
# message per call with only `content` and `tool_call_id`. The only append group
# the loop ever produces is that tool group, so that is the only append tested.
#
# Compared: today's path (bounded render of the group + two full-history renders
# for the generation prompt) vs rendering the group once with
# add_generation_prompt=True. Upstream main, real tokenizers, token ids compared.
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
    ("gemma4", ["google/gemma-4-27b-it", "unsloth/gemma-4-27b-it", "unsloth/gemma-4-12b-it", "unsloth/gemma-4-4b-it",
                "google/gemma-4-12b-it", "unsloth/gemma-3-27b-it"]),
    ("gptoss", ["openai/gpt-oss-20b"]),
    ("deepseek", ["deepseek-ai/DeepSeek-V3.2-Exp", "deepseek-ai/DeepSeek-V3"]),
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
TOOL_RESULTS = [
    "Pittsburgh had 302,971 residents at the latest census estimate.",
    "Second lookup: 1,244,000 in the metro area.",
]


def assistant_turn(turn, calls):
    """What tool_agent_loop._build_assistant_message produces for `calls` tool calls."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "lookup", "arguments": {"q": f"{turn}-{index}"}},
                "id": f"call-{turn}-{index}",
            }
            for index in range(calls)
        ],
    }


def tool_group(turn, calls):
    """What tool_agent_loop appends after that assistant turn: one tool message per call."""
    return [
        {"role": "tool", "content": TOOL_RESULTS[index % len(TOOL_RESULTS)], "tool_call_id": f"call-{turn}-{index}"}
        for index in range(calls)
    ]


def trajectory(system_prompt, prior_turns, calls):
    """(previous_messages, appended_groups) for the incremental call at turn `prior_turns`."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": "You are a helpful agent. Use tools to answer."})
    messages.append({"role": "user", "content": "Find the population of Pittsburgh and its ten largest suburbs."})
    for turn in range(prior_turns):
        messages.append(assistant_turn(turn, 1))
        messages.extend(tool_group(turn, 1))
    messages.append(assistant_turn(prior_turns, calls))
    return messages, [tool_group(prior_turns, calls)]


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
result = {"family": family, "model": model, "cases": [], "arg_form": "dict (as tool_agent_loop passes it)"}

# The same 24 cases for every model. Templates without an enable_thinking switch
# ignore the kwarg, so their two halves are the same renders; that is reported.
kwargs_variants = [{}, {"enable_thinking": False}]
result["template_has_thinking_switch"] = "enable_thinking" in (getattr(tok, "chat_template", "") or "")

for kwargs in kwargs_variants:
    for system_prompt in (True, False):
        for prior_turns in (0, 10, 50):
            for calls in (1, 2):
                case = {"kwargs": kwargs, "tools": True, "system_prompt": system_prompt,
                        "prior_turns": prior_turns, "append": f"{calls} tool response(s)"}
                previous, groups = trajectory(system_prompt, prior_turns, calls)
                appended = [message for group in groups for message in group]
                updated = previous + appended
                try:
                    current = builder_cls(tok, chat_template_kwargs=dict(kwargs)).tokenize_non_assistant_incremental_messages(
                        previous, updated, tools=TOOLS)
                except Exception as exc:  # noqa: BLE001
                    case["current_error"] = f"{type(exc).__name__}: {str(exc)[:90]}"
                try:
                    truth = builder_cls(tok, chat_template_kwargs=dict(kwargs)).render_delta_token_id(
                        previous, appended, add_generation_prompt=True, tools=TOOLS)
                except Exception as exc:  # noqa: BLE001
                    truth = None
                    case["truth_error"] = f"{type(exc).__name__}: {str(exc)[:90]}"
                try:
                    folded, calls_made = folded_incremental(
                        builder_cls(tok, chat_template_kwargs=dict(kwargs)), previous, updated, TOOLS, len(groups))
                    case["template_renders"] = calls_made
                except Exception as exc:  # noqa: BLE001
                    case["folded_error"] = f"{type(exc).__name__}: {str(exc)[:90]}"
                if "current_error" not in case and "folded_error" not in case:
                    case["same"] = folded == current
                    if truth is not None:
                        case["current_matches_truth"] = current == truth
                        case["folded_matches_truth"] = folded == truth
                    if not case["same"]:
                        case["current_text"] = tok.decode(current)
                        case["folded_text"] = tok.decode(folded)
                        if truth is not None:
                            case["truth_text"] = tok.decode(truth)
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
    print(f"[fold] upstream main @{head}, harness-shaped tool trajectories")
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
            with_truth = [c for c in cases if "current_matches_truth" in c]
            truth_note = (
                f" truth: current=={sum(c['current_matches_truth'] for c in with_truth)}/{len(with_truth)} "
                f"folded=={sum(c['folded_matches_truth'] for c in with_truth)}/{len(with_truth)}"
                if with_truth else " truth: full-history diff not prefix-stable"
            )
            print(
                f"[FOLD] {family:<10} {model:<30} thinking_switch={data.get('template_has_thinking_switch')} "
                f"same={same}/{len(cases)} differ={len(differ)} not_rendered={len(errors)}{truth_note}"
            )
            for c in differ:
                print(f"[DIFF] {family} kwargs={c['kwargs']} system={c['system_prompt']} prior={c['prior_turns']} append={c['append']}")
                print(f"[DIFF]    current: {c['current_text'][-160:]!r}")
                print(f"[DIFF]    folded : {c['folded_text'][-160:]!r}")
                if "truth_text" in c:
                    print(f"[DIFF]    truth  : {c['truth_text'][-160:]!r}")
            seen = set()
            for c in errors:
                key = (c.get("current_error"), c.get("folded_error"))
                if key in seen:
                    continue
                seen.add(key)
                which = "both" if c.get("current_error") and c.get("folded_error") else ("current" if c.get("current_error") else "folded")
                print(f"[ERR] {family} system={c['system_prompt']} prior={c['prior_turns']} append={c['append']} {which}: {c.get('current_error') or c.get('folded_error')}")
            for text in texts[:3]:
                print(f"[TEXT] {family}: {text[-90:]!r}")
            break
    print("FOLD_HARNESS_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
