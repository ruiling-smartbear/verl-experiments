# verl #7628: does the merged "fuse the generation prompt into the final append
# group" path reproduce the token ids of the previous full-history path?
#
# Two checkouts of verl - the merge commit (3dab856) and its parent (8e4a572) -
# driven through the public tokenize_non_assistant_incremental_messages API on
# trajectories shaped exactly like tool_agent_loop's. Token ids compared case by
# case; where the template is prefix stable, the naive full-history suffix diff
# is compared as an independent reference too. Tokenizer files only, CPU.
#
# Only the tokenizer package of each checkout is imported (parent packages are
# stubbed), so a bare venv with transformers is enough - no torch, no ray.
import json
import os
import subprocess
import sys
import tempfile

UPSTREAM_REPO = "https://github.com/verl-project/verl"
BEFORE_SHA = "8e4a572980c4e9a894dc5bb2fd4976652207bdaa"  # parent of the #7628 merge
AFTER_SHA = "3dab856da5eef3a5d20bc219371a9bcbf1aaa5a6"  # the #7628 merge commit
# (builder family, Hub model, variants). Variant = how the harness encodes
# tool-call arguments (dict = what OpenAIFunctionCallSchema.model_dump() gives,
# json = a JSON string) and, with "+strsyn", whether verl's synthetic assistant
# message is patched to carry `"arguments": "{}"` instead of `{}`.
FAMILY_MODELS = [
    ("qwen", "Qwen/Qwen2-7B-Instruct", ["dict"]),
    ("qwen25", "Qwen/Qwen2.5-7B-Instruct", ["dict"]),
    ("qwen3", "Qwen/Qwen3-8B", ["dict"]),
    ("qwen35", "Qwen/Qwen3.5-9B", ["dict"]),
    ("minimaxm2", "MiniMaxAI/MiniMax-M2", ["dict"]),
    ("glm47", "zai-org/GLM-4.7", ["dict"]),
    ("gemma4", "google/gemma-4-12b-it", ["dict"]),
    ("gptoss", "openai/gpt-oss-20b", ["dict"]),
    ("deepseek", "deepseek-ai/DeepSeek-V3", ["dict", "json", "dict+strsyn", "json+strsyn"]),
    ("deepseek", "deepseek-ai/DeepSeek-R1", ["dict", "json", "dict+strsyn", "json+strsyn"]),
    ("deepseek", "deepseek-ai/DeepSeek-V3.1", ["dict", "json", "dict+strsyn", "json+strsyn"]),
    ("deepseek", "deepseek-ai/DeepSeek-V3.2-Exp", ["dict", "json", "dict+strsyn", "json+strsyn"]),
    ("default", "Qwen/Qwen3-8B", ["dict"]),
]

RUNNER = r'''
import json, sys, types
verl_path, family, model, variant, out = sys.argv[1:6]
arg_form, _, patch = variant.partition("+")
for name, sub in (("verl", "verl"), ("verl.utils", "verl/utils")):
    stub = types.ModuleType(name)
    stub.__path__ = [f"{verl_path}/{sub}"]
    sys.modules[name] = stub
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
TOOL_RESULTS = [
    "Pittsburgh had 302,971 residents at the latest census estimate.",
    "Second lookup: 1,244,000 in the metro area.",
]


def arguments(turn, index):
    payload = {"q": f"{turn}-{index}"}
    return json.dumps(payload) if arg_form == "json" else payload


def assistant_turn(turn, calls):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"type": "function", "function": {"name": "lookup", "arguments": arguments(turn, index)}, "id": f"call-{turn}-{index}"}
            for index in range(calls)
        ],
    }


def tool_group(turn, calls):
    return [
        {"role": "tool", "content": TOOL_RESULTS[index % len(TOOL_RESULTS)], "tool_call_id": f"call-{turn}-{index}"}
        for index in range(calls)
    ]


def trajectory(system_prompt, prior_turns, calls):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": "You are a helpful agent. Use tools to answer."})
    messages.append({"role": "user", "content": "Find the population of Pittsburgh and its ten largest suburbs."})
    for turn in range(prior_turns):
        messages.append(assistant_turn(turn, 1))
        messages.extend(tool_group(turn, 1))
    messages.append(assistant_turn(prior_turns, calls))
    return messages, tool_group(prior_turns, calls)


tok = AutoTokenizer.from_pretrained(model)
builder_cls = get_continuous_token_builder_class(family)
if patch == "strsyn":
    original = builder_cls._synthetic_assistant_for_tools

    def _string_arguments(self, tool_messages):
        message = original(self, tool_messages)
        for call in message["tool_calls"]:
            call["function"]["arguments"] = "{}"
        return message

    builder_cls._synthetic_assistant_for_tools = _string_arguments
cases = []
for kwargs in ({}, {"enable_thinking": False}):
    for system_prompt in (True, False):
        for prior_turns in (0, 10, 50):
            for calls in (1, 2):
                case = {"kwargs": kwargs, "system_prompt": system_prompt, "prior_turns": prior_turns, "calls": calls}
                previous, appended = trajectory(system_prompt, prior_turns, calls)
                try:
                    builder = builder_cls(tok, chat_template_kwargs=dict(kwargs))
                    case["ids"] = builder.tokenize_non_assistant_incremental_messages(previous, previous + appended, tools=TOOLS)
                    case["text"] = tok.decode(case["ids"])
                except Exception as exc:  # noqa: BLE001
                    case["error"] = f"{type(exc).__name__}: {str(exc)[:90]}"
                try:
                    case["reference"] = builder_cls(tok, chat_template_kwargs=dict(kwargs)).render_delta_token_id(
                        previous, appended, add_generation_prompt=True, tools=TOOLS)
                except Exception as exc:  # noqa: BLE001
                    case["reference_error"] = f"{type(exc).__name__}: {str(exc)[:90]}"
                cases.append(case)
with open(out, "w") as fh:
    json.dump({"has_thinking_switch": "enable_thinking" in (getattr(tok, "chat_template", "") or ""), "cases": cases}, fh)
'''


def checkout(path: str, sha: str) -> str:
    if not os.path.isdir(os.path.join(path, "verl")):
        subprocess.run(
            f"rm -rf {path} && mkdir -p {path} && cd {path} && git init -q && git remote add origin {UPSTREAM_REPO} "
            f"&& git fetch -q --depth 1 origin {sha} && git checkout -q FETCH_HEAD",
            shell=True, check=True,
        )
    return subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"], text=True).strip()


def run(verl_path: str, family: str, model: str, variant: str):
    out = tempfile.mktemp(suffix=".json")
    proc = subprocess.run(
        [sys.executable, "-c", RUNNER, verl_path, family, model, variant, out], capture_output=True, text=True
    )
    if proc.returncode != 0:
        return ((proc.stderr or "").strip().splitlines() or [""])[-1][:120]
    with open(out) as fh:
        return json.load(fh)


def main() -> int:
    import transformers

    before = checkout("/tmp/verl_before", BEFORE_SHA)
    after = checkout("/tmp/verl_after", AFTER_SHA)
    print(f"[fused] verl @{before} (before #7628) vs @{after} (the #7628 merge), transformers {transformers.__version__}, "
          f"tool_agent_loop-shaped trajectories, 24 cases per row")
    for family, model, variants in FAMILY_MODELS:
        for variant in variants:
            old = run("/tmp/verl_before", family, model, variant)
            if isinstance(old, str):
                print(f"[skip] {family} {model} {variant}: {old}")
                continue
            new = run("/tmp/verl_after", family, model, variant)
            if isinstance(new, str):
                print(f"[skip] {family} {model} {variant} (after): {new}")
                continue
            same = differ = only_before = only_after = neither = ref_old = ref_new = with_ref = 0
            first_diff = None
            errors = set()
            for o, n in zip(old["cases"], new["cases"]):
                if "ids" in o and "ids" in n:
                    if o["ids"] == n["ids"]:
                        same += 1
                    else:
                        differ += 1
                        first_diff = first_diff or (o, n)
                elif "ids" in o:
                    only_before += 1
                elif "ids" in n:
                    only_after += 1
                else:
                    neither += 1
                if "ids" in n and "reference" in n:
                    with_ref += 1
                    ref_old += o.get("ids") == n["reference"]
                    ref_new += n["ids"] == n["reference"]
                if "error" in o or "error" in n:
                    errors.add((o.get("error"), n.get("error")))
            ref_note = f"ref: before=={ref_old}/{with_ref} after=={ref_new}/{with_ref}" if with_ref else "ref: n/a"
            print(
                f"[FUSED] {family:<9} {model:<30} {variant:<12} thinking_switch={new['has_thinking_switch']} "
                f"after==before {same}/24 differ={differ} only_before={only_before} only_after={only_after} "
                f"neither={neither} | {ref_note}"
            )
            if first_diff:
                o, n = first_diff
                print(f"[DIFF]  {family} {model} {variant} system={o['system_prompt']} prior={o['prior_turns']} calls={o['calls']} kwargs={o['kwargs']}")
                print(f"[DIFF]    before: {o['text'][-170:]!r}")
                print(f"[DIFF]    after : {n['text'][-170:]!r}")
            if only_after:
                n = next(c for c in new["cases"] if "ids" in c)
                print(f"[TAIL]  {family} {model} {variant} after: {n['text'][-170:]!r}")
            for err in sorted(errors, key=str)[:2]:
                print(f"[ERR]   {family} {model} {variant} before={err[0]} after={err[1]}")
    print("FUSED_CHECK_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
