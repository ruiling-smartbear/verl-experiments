# verl #7637: validate the two candidate fixes for the DeepSeek-R1 distill
# builder-construction regression, each applied alone on upstream main.
#
#   fix1 = fall back to the default builder when the inferred builder's required
#          token is missing (+ base synthetic tool-call arguments as a JSON string)
#   fix2 = resolve the family from an exact tokenizer special-token key and use
#          the DeepSeek builder for these checkpoints
#
# Parts:
#   [ID]  identity: the 13-row / 24-case tool-append harness (same trajectories as
#         verl_fused_check.py) must stay byte-identical between main and each fix
#         for every model that works on main.
#   [SEL] selection: what create_continuous_token_builder() returns (or raises)
#         on main / fix1 / fix2 for the distills and control models.
#   [RND] render: the factory-built builder's first / third tool-turn appends vs
#         the template's full-conversation render.
#   [KEY] the exact special-token key inputs per tokenizer.
import json
import os
import subprocess
import sys
import tempfile

UPSTREAM_REPO = "https://github.com/verl-project/verl"
MAIN_SHA = "9c76436b93a2fc4d24f9d17d9265b3b467a30f4f"  # upstream main, 2026-08-31
PATCHES = "https://raw.githubusercontent.com/ruiling-smartbear/verl-experiments/e93bbc014e7d983698194b89c7c5cbb8101d54d6/patches/"
FIX1_PATCH = PATCHES + "0001-rollout-fix-fall-back-to-the-default-continuous-toke.patch"
FIX2_PATCH = PATCHES + "0001-rollout-fix-resolve-the-continuous-token-family-from.patch"

FAMILY_MODELS = [
    ("qwen", "Qwen/Qwen2-7B-Instruct", ["dict"]),
    ("qwen25", "Qwen/Qwen2.5-7B-Instruct", ["dict"]),
    ("qwen3", "Qwen/Qwen3-8B", ["dict"]),
    ("qwen35", "Qwen/Qwen3.5-9B", ["dict"]),
    ("minimaxm2", "MiniMaxAI/MiniMax-M2", ["dict"]),
    ("glm47", "zai-org/GLM-4.7", ["dict"]),
    ("gemma4", "google/gemma-4-12b-it", ["dict"]),
    ("gptoss", "openai/gpt-oss-20b", ["dict"]),
    ("deepseek", "deepseek-ai/DeepSeek-V3", ["dict", "json"]),
    ("deepseek", "deepseek-ai/DeepSeek-R1", ["dict", "json"]),
    ("deepseek", "deepseek-ai/DeepSeek-V3.1", ["dict", "json"]),
    ("deepseek", "deepseek-ai/DeepSeek-V3.2-Exp", ["dict", "json"]),
    ("default", "Qwen/Qwen3-8B", ["dict"]),
]

# (model, tokenizer class override). DeepSeek-R1-0528-Qwen3-8B decodes wrongly as
# the LlamaTokenizerFast transformers 5 picks for it, so load the Qwen2 tokenizer
# it actually is; ids are unaffected, only decode-for-display is.
FACTORY_MODELS = [
    ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "Qwen2TokenizerFast"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", ""),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", ""),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", ""),
    ("Qwen/Qwen3-8B", ""),
    ("Qwen/Qwen2.5-7B-Instruct", ""),
    ("deepseek-ai/DeepSeek-V3.1", ""),
]

IDENTITY_RUNNER = r'''
import json, sys, types
verl_path, family, model, variant, out = sys.argv[1:6]
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
    return json.dumps(payload) if variant == "json" else payload


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
                cases.append(case)
with open(out, "w") as fh:
    json.dump(cases, fh)
'''

FACTORY_RUNNER = r'''
import json, sys, types
verl_path, model, tokenizer_class, out = sys.argv[1:5]
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
    builder = create_continuous_token_builder(tok, hf_model_type=model_type)
    result["builder"] = type(builder).__name__
except Exception as exc:  # noqa: BLE001
    result["error"] = f"{type(exc).__name__}: {str(exc)[:90]}"
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
        result["renders"].append({"turn": turns, "text": tok.decode(ids), "matches_template": ids == reference})
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


def token_state(tok, token):
    token_id = tok.convert_tokens_to_ids(token)
    if token_id is None or token_id == getattr(tok, "unk_token_id", None):
        return f"absent({token_id})"
    return str(token_id)


def main():
    import transformers
    from transformers import AutoTokenizer

    trees = {
        "main": checkout("/tmp/verl_dc_main", MAIN_SHA),
        "fix1": checkout("/tmp/verl_dc_fix1", MAIN_SHA, FIX1_PATCH),
        "fix2": checkout("/tmp/verl_dc_fix2", MAIN_SHA, FIX2_PATCH),
    }
    print(f"[trees] transformers {transformers.__version__} | " + " | ".join(f"{k}: {v}" for k, v in trees.items()))

    print("\n--- [ID] identity on models that work on main (24 tool-append cases per row)")
    for family, model, variants in FAMILY_MODELS:
        for variant in variants:
            base = run(IDENTITY_RUNNER, ["/tmp/verl_dc_main", family, model, variant])
            if isinstance(base, str):
                print(f"[skip] {family} {model} {variant}: {base}")
                continue
            row = f"[ID]  {family:<9} {model:<33} {variant:<5}"
            for tree_name, tree_path in (("fix1", "/tmp/verl_dc_fix1"), ("fix2", "/tmp/verl_dc_fix2")):
                fixed = run(IDENTITY_RUNNER, [tree_path, family, model, variant])
                if isinstance(fixed, str):
                    row += f" {tree_name}: RUNNER FAILED {fixed}"
                    continue
                same = sum(1 for b, f in zip(base, fixed) if b.get("ids") == f.get("ids") and b.get("error") == f.get("error"))
                row += f" {tree_name}=={'main' if same == len(base) else 'DIFFERS'} ({same}/{len(base)})"
            print(row)

    print("\n--- [SEL]/[RND] factory selection and renders")
    for model, tokenizer_class in FACTORY_MODELS:
        for tree_name, tree_path in (("main", "/tmp/verl_dc_main"), ("fix1", "/tmp/verl_dc_fix1"), ("fix2", "/tmp/verl_dc_fix2")):
            res = run(FACTORY_RUNNER, [tree_path, model, tokenizer_class])
            if isinstance(res, str):
                print(f"[SEL] {model:<42} {tree_name}: RUNNER FAILED {res}")
                continue
            if "error" in res:
                print(f"[SEL] {model:<42} {tree_name}: model_type={res['model_type']} -> {res['error']}")
                continue
            print(f"[SEL] {model:<42} {tree_name}: model_type={res['model_type']} -> {res['builder']}")
            for render in res["renders"]:
                if "error" in render:
                    print(f"[RND]   turn {render['turn']}: {render['error']}")
                else:
                    print(f"[RND]   turn {render['turn']}: == template: {render['matches_template']} | {render['text'][-110:]!r}")

    print("\n--- [KEY] special-token key inputs (id, or absent(raw))")
    key_models = [m for m, _ in FACTORY_MODELS] + ["deepseek-ai/DeepSeek-R1"]
    for model in key_models:
        try:
            tok = AutoTokenizer.from_pretrained(model)
        except Exception as exc:  # noqa: BLE001
            print(f"[KEY] {model:<42} tokenizer load failed: {type(exc).__name__}")
            continue
        eos = token_state(tok, "<｜end▁of▁sentence｜>")
        assistant = token_state(tok, "<｜Assistant｜>")
        im_end = token_state(tok, "<|im_end|>")
        fires = not eos.startswith("absent") and not assistant.startswith("absent") and im_end.startswith("absent")
        print(f"[KEY] {model:<42} eos={eos:<14} assistant={assistant:<14} im_end={im_end:<14} key_fires={fires}")
    print("\nDISTILL_FIX_CHECK_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
