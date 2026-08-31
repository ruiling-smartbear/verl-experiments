# Is the generation-prompt delta cacheable across turns? (verl #7617)
#
# For every text Continuous Token builder family, this runs upstream main's
# full-history `_tokenize_generation_prompt_delta` over a growing multi-turn
# tool rollout with a real tokenizer and chat template, and reports:
#   - does the delta stay constant as the history grows (per final role)?
#   - is it the same for tool / user / system final messages?
#   - does the bounded pseudo-conversation render reproduce it?
# Tokenization only, CPU only.
import json
import os
import subprocess
import sys
import tempfile

UPSTREAM_REPO = "https://github.com/verl-project/verl"

# One or more Hub candidates per family; the first that loads wins.
FAMILY_MODELS = {
    "qwen25": ["Qwen/Qwen2.5-7B-Instruct"],
    "qwen3": ["Qwen/Qwen3-8B"],
    "qwen35": ["Qwen/Qwen3.5-9B"],
    "minimaxm2": ["MiniMaxAI/MiniMax-M2", "MiniMaxAI/MiniMax-Text-01"],
    "glm47": ["zai-org/GLM-4.7", "zai-org/GLM-4.6", "zai-org/GLM-4.5"],
    "glm5": ["zai-org/GLM-5"],
    "gemma4": ["google/gemma-4-27b-it", "google/gemma-3-27b-it"],
    "gptoss": ["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
    "deepseek": ["deepseek-ai/DeepSeek-V3.2", "deepseek-ai/DeepSeek-V3.1", "deepseek-ai/DeepSeek-V3"],
    "deepseekv4": ["deepseek-ai/DeepSeek-V4"],
    "default": ["Qwen/Qwen3-8B"],
}
TURNS = 20

RUNNER = r'''
import json, sys
verl_path, family, model, turns, out = sys.argv[1:6]
sys.path.insert(0, verl_path)
from transformers import AutoTokenizer
from verl.utils.tokenizer.continuous_token import (
    _SYNTHETIC_SYSTEM_MESSAGE,
    _SYNTHETIC_USER_MESSAGE,
)
from verl.utils.tokenizer.continuous_token_wiring import get_continuous_token_builder_class

TOOLS = [{
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Look up a population figure.",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
    },
}]
TOOL_RESULT = ("The lookup returned 302,971 residents as of the latest census estimate. " * 6)[:400]
USER_TAIL = {"role": "user", "content": "Thanks, keep going."}
SYSTEM_TAIL = {"role": "system", "content": "Budget: three more tool calls."}


def bounded(builder, messages, tools):
    """The pseudo-conversation render this PR uses for the Qwen builders."""
    last = messages[-1]
    if last.get("role") == "tool":
        prefix = [
            _SYNTHETIC_SYSTEM_MESSAGE,
            _SYNTHETIC_USER_MESSAGE,
            builder._synthetic_assistant_for_tools([last]),
            last,
        ]
    else:
        prefix = [_SYNTHETIC_SYSTEM_MESSAGE, _SYNTHETIC_USER_MESSAGE, last]
    return builder.render_delta_token_id(prefix, [], add_generation_prompt=True, tools=tools)


tok = AutoTokenizer.from_pretrained(model)
builder_cls = get_continuous_token_builder_class(family)
result = {"family": family, "model": model, "cases": {}}
for tools in (None, TOOLS):
    builder = builder_cls(tok)
    messages = [
        {"role": "system", "content": "You are a helpful agent. Use tools to answer."},
        {"role": "user", "content": "Find the population of Pittsburgh and its ten largest suburbs."},
    ]
    seen = {"tool": [], "user": [], "system": []}
    bounded_matches = {"tool": [], "user": [], "system": []}
    for turn in range(int(turns)):
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"call-{turn}", "type": "function", "function": {"name": "lookup", "arguments": {"q": str(turn)}}}]})
        messages.append({"role": "tool", "content": TOOL_RESULT, "tool_call_id": f"call-{turn}", "name": "lookup"})
        for role, history in (
            ("tool", messages),
            ("user", messages + [USER_TAIL]),
            ("system", messages + [SYSTEM_TAIL]),
        ):
            try:
                full = builder._tokenize_generation_prompt_delta(history, tools=tools)
            except Exception as exc:  # noqa: BLE001
                full = f"ERROR {type(exc).__name__}: {str(exc)[:80]}"
            seen[role].append(full)
            try:
                bounded_matches[role].append(bounded(builder, history, tools) == full)
            except Exception as exc:  # noqa: BLE001
                bounded_matches[role].append(f"ERROR {type(exc).__name__}")
    key = "tools" if tools else "no-tools"
    result["cases"][key] = {
        role: {
            "constant_across_turns": len({json.dumps(v) for v in seen[role]}) == 1,
            "distinct_values": len({json.dumps(v) for v in seen[role]}),
            "first": seen[role][0],
            "last": seen[role][-1],
            "bounded_matches_all": all(v is True for v in bounded_matches[role]),
        }
        for role in seen
    }
    roles = result["cases"][key]
    result["cases"][key]["same_across_roles"] = len(
        {json.dumps(roles[role]["last"]) for role in ("tool", "user", "system")}
    ) == 1
with open(out, "w") as fh:
    json.dump(result, fh)
'''


def clone(repo: str, path: str) -> str:
    if not os.path.isdir(os.path.join(path, "verl")):
        subprocess.run(f"rm -rf {path} && git clone -q --depth 1 {repo} {path}", shell=True, check=True)
    return subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"], text=True).strip()


def main() -> int:
    head = clone(UPSTREAM_REPO, "/tmp/verl_main")
    print(f"[cache] upstream main @{head}, {TURNS} tool turns per family")
    for family, candidates in FAMILY_MODELS.items():
        for model in candidates:
            out = tempfile.mktemp(suffix=".json")
            proc = subprocess.run(
                [sys.executable, "-c", RUNNER, "/tmp/verl_main", family, model, str(TURNS), out],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                tail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
                print(f"[skip] {family} {model}: {tail[0][:110]}")
                continue
            with open(out) as fh:
                data = json.load(fh)
            for case, roles in data["cases"].items():
                same = roles.pop("same_across_roles")
                summary = " ".join(
                    f"{role}:const={roles[role]['constant_across_turns']}"
                    f"/values={roles[role]['distinct_values']}"
                    f"/bounded={roles[role]['bounded_matches_all']}"
                    for role in ("tool", "user", "system")
                )
                print(f"[CACHE] {family:<10} {model:<32} {case:<8} {summary} same_across_roles={same}")
            break
    print("CACHE_CHECK_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
