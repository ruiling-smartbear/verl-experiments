# verl #7637, option 3: data.continuous_token.model_family override + an actionable
# construction error, applied alone on upstream main.
#
#   [ID]  identity: the 13-row / 24-case tool-append harness must stay byte-identical
#         between main and main+patch for every model that works on main (the patch
#         touches no successful path).
#   [SEL] selection: what create_continuous_token_builder() does on main / patched for
#         the distills and controls — auto (error text before/after) and, on the patched
#         tree, model_family="deepseek".
#   [RND] render: the explicitly selected builder's first / third tool-turn appends vs
#         the template's full-conversation render.
import json
import os
import subprocess
import sys
import tempfile

UPSTREAM_REPO = "https://github.com/verl-project/verl"
MAIN_SHA = "9c7643648f1e1e109dd30da2da3cf94e1317b229"  # upstream main, 2026-08-31
PATCH = (
    "https://raw.githubusercontent.com/ruiling-smartbear/verl-experiments/2a7716dc38e905e3aa50f76e0a3e130d7c3f3c7a/"
    "patches/0001-rollout-feat-explicit-continuous-token-model-family.patch"
)

FAMILY_MODELS = [
    ("qwen", "Qwen/Qwen2-7B-Instruct"),
    ("qwen25", "Qwen/Qwen2.5-7B-Instruct"),
    ("qwen3", "Qwen/Qwen3-8B"),
    ("qwen35", "Qwen/Qwen3.5-9B"),
    ("minimaxm2", "MiniMaxAI/MiniMax-M2"),
    ("glm47", "zai-org/GLM-4.7"),
    ("gemma4", "google/gemma-4-12b-it"),
    ("gptoss", "openai/gpt-oss-20b"),
    ("deepseek", "deepseek-ai/DeepSeek-V3"),
    ("deepseek", "deepseek-ai/DeepSeek-R1"),
    ("deepseek", "deepseek-ai/DeepSeek-V3.1"),
    ("deepseek", "deepseek-ai/DeepSeek-V3.2-Exp"),
    ("default", "Qwen/Qwen3-8B"),
]

FACTORY_MODELS = [
    ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "Qwen2TokenizerFast"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", ""),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", ""),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", ""),
    ("Qwen/Qwen3-8B", ""),
    ("deepseek-ai/DeepSeek-V3.1", ""),
]

IDENTITY_RUNNER = r'''
import json, sys, types
verl_path, family, model, out = sys.argv[1:5]
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


def assistant_turn(turn, calls):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"type": "function", "function": {"name": "lookup", "arguments": {"q": f"{turn}-{index}"}}, "id": f"call-{turn}-{index}"}
            for index in range(calls)
        ],
    }


def tool_group(turn, calls):
    return [
        {"role": "tool", "content": TOOL_RESULTS[index % len(TOOL_RESULTS)], "tool_call_id": f"call-{turn}-{index}"}
        for index in range(calls)
    ]


tok = AutoTokenizer.from_pretrained(model)
builder_cls = get_continuous_token_builder_class(family)
cases = []
for kwargs in ({}, {"enable_thinking": False}):
    for system_prompt in (True, False):
        for prior_turns in (0, 10, 50):
            for calls in (1, 2):
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": "You are a helpful agent. Use tools to answer."})
                messages.append({"role": "user", "content": "Find the population of Pittsburgh and its ten largest suburbs."})
                for turn in range(prior_turns):
                    messages.append(assistant_turn(turn, 1))
                    messages.extend(tool_group(turn, 1))
                messages.append(assistant_turn(prior_turns, calls))
                appended = tool_group(prior_turns, calls)
                case = {}
                try:
                    builder = builder_cls(tok, chat_template_kwargs=dict(kwargs))
                    case["ids"] = builder.tokenize_non_assistant_incremental_messages(messages, messages + appended, tools=TOOLS)
                except Exception as exc:  # noqa: BLE001
                    case["error"] = f"{type(exc).__name__}: {str(exc)[:90]}"
                cases.append(case)
with open(out, "w") as fh:
    json.dump(cases, fh)
'''

FACTORY_RUNNER = r'''
import json, sys, types
verl_path, model, tokenizer_class, family, out = sys.argv[1:6]
for name, sub in (("verl", "verl"), ("verl.utils", "verl/utils")):
    stub = types.ModuleType(name)
    stub.__path__ = [f"{verl_path}/{sub}"]
    sys.modules[name] = stub
import transformers
from transformers import AutoConfig, AutoTokenizer
from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

loader = getattr(transformers, tokenizer_class) if tokenizer_class else AutoTokenizer
tok = loader.from_pretrained(model)
model_type = AutoConfig.from_pretrained(model).model_type
result = {"model_type": model_type}
try:
    builder = create_continuous_token_builder(tok, model_family=family, hf_model_type=model_type)
    result["builder"] = type(builder).__name__
except Exception as exc:  # noqa: BLE001
    result["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    with open(out, "w") as fh:
        json.dump(result, fh)
    raise SystemExit(0)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "d",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
    },
}]


def turn(i):
    return {
        "role": "assistant",
        "content": "<think>\nx\n</think>\n\n",
        "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "lookup", "arguments": {"q": str(i)}}}],
    }


def tool(i):
    return {"role": "tool", "content": f"answer {i}", "tool_call_id": f"c{i}"}


def with_string_arguments(messages):
    out_messages = []
    for message in messages:
        message = dict(message)
        if message.get("tool_calls"):
            message["tool_calls"] = [
                {**call, "function": {**call["function"], "arguments": json.dumps(call["function"]["arguments"])}}
                for call in message["tool_calls"]
            ]
        out_messages.append(message)
    return out_messages


result["renders"] = []
for turns in (1, 3):
    previous = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
    for i in range(turns - 1):
        previous += [turn(i), tool(i)]
    previous.append(turn(turns - 1))
    appended = [tool(turns - 1)]
    try:
        full = tok.apply_chat_template(with_string_arguments(previous + appended), tokenize=True,
                                       add_generation_prompt=True, return_dict=False)
        prefix = tok.apply_chat_template(with_string_arguments(previous), tokenize=True,
                                         add_generation_prompt=False, return_dict=False)
        reference = full[len(prefix):] if full[: len(prefix)] == prefix else None
        ids = builder.tokenize_non_assistant_incremental_messages(previous, previous + appended, tools=TOOLS)
        result["renders"].append({"turn": turns, "matches_template": ids == reference})
    except Exception as exc:  # noqa: BLE001
        result["renders"].append({"turn": turns, "error": f"{type(exc).__name__}: {str(exc)[:90]}"})
with open(out, "w") as fh:
    json.dump(result, fh)
'''


def checkout(path, sha, patch_url=None):
    if not os.path.isdir(os.path.join(path, "verl")):
        subprocess.run(
            f"rm -rf {path} && mkdir -p {path} && cd {path} && git init -q && git remote add origin {UPSTREAM_REPO} "
            f"&& git fetch -q --depth 1 origin {sha} && git checkout -q FETCH_HEAD",
            shell=True, check=True,
        )
        if patch_url:
            subprocess.run(
                f"cd {path} && curl -sL '{patch_url}' -o /tmp/fix.patch && git apply --index /tmp/fix.patch",
                shell=True, check=True,
            )
    head = subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"], text=True).strip()
    changed = subprocess.check_output(["git", "-C", path, "diff", "--cached", "--stat"], text=True).strip().splitlines()
    return head + (f" + patch ({changed[-1].strip()})" if changed else "")


def run(runner, args):
    out = tempfile.mktemp(suffix=".json")
    proc = subprocess.run([sys.executable, "-c", runner, *args, out], capture_output=True, text=True)
    if proc.returncode != 0:
        return ((proc.stderr or "").strip().splitlines() or [""])[-1][:140]
    with open(out) as fh:
        return json.load(fh)


def main():
    import transformers

    trees = {
        "main": checkout("/tmp/verl_fc_main", MAIN_SHA),
        "fix3": checkout("/tmp/verl_fc_fix3", MAIN_SHA, PATCH),
    }
    print(f"[trees] transformers {transformers.__version__} | " + " | ".join(f"{k}: {v}" for k, v in trees.items()))

    print("\n--- [ID] identity on models that work on main (24 tool-append cases per row)")
    for family, model in FAMILY_MODELS:
        base = run(IDENTITY_RUNNER, ["/tmp/verl_fc_main", family, model])
        if isinstance(base, str):
            print(f"[skip] {family} {model}: {base}")
            continue
        fixed = run(IDENTITY_RUNNER, ["/tmp/verl_fc_fix3", family, model])
        if isinstance(fixed, str):
            print(f"[ID]  {family:<9} {model:<33} RUNNER FAILED {fixed}")
            continue
        same = sum(1 for b, f in zip(base, fixed) if b.get("ids") == f.get("ids") and b.get("error") == f.get("error"))
        print(f"[ID]  {family:<9} {model:<33} fix3=={'main' if same == len(base) else 'DIFFERS'} ({same}/{len(base)})")

    print("\n--- [SEL]/[RND] factory behavior (auto on both trees; model_family=deepseek on the patched tree)")
    for model, tokenizer_class in FACTORY_MODELS:
        for tree_name, tree_path, family in (
            ("main auto", "/tmp/verl_fc_main", "auto"),
            ("fix3 auto", "/tmp/verl_fc_fix3", "auto"),
            ("fix3 deepseek", "/tmp/verl_fc_fix3", "deepseek"),
        ):
            if family == "deepseek" and "Distill" not in model and "0528" not in model:
                continue
            res = run(FACTORY_RUNNER, [tree_path, model, tokenizer_class, family])
            if isinstance(res, str):
                print(f"[SEL] {model:<42} {tree_name}: RUNNER FAILED {res}")
                continue
            if "error" in res:
                guided = "data.continuous_token.model_family" in res["error"] and "deepseek" in res["error"]
                print(f"[SEL] {model:<42} {tree_name}: model_type={res['model_type']} -> ERROR guided={guided}")
                print(f"        {res['error'][:220]}")
                continue
            renders = " ".join(
                f"turn{r['turn']}=={r.get('matches_template', 'ERR')}" for r in res["renders"]
            )
            print(f"[SEL] {model:<42} {tree_name}: model_type={res['model_type']} -> {res['builder']} | {renders}")
    print("\nFAMILY_CONFIG_CHECK_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
