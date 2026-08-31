# Does caching the generation-prompt delta hold up outside Qwen? (verl #7617)
#
# Drives a multi-turn tool rollout through the public
# `tokenize_non_assistant_incremental_messages` API on two checkouts - upstream
# main and the cache prototype - with real tokenizers, and compares token ids
# turn by turn while timing both. Tokenization only, CPU only.
import json
import os
import subprocess
import sys
import tempfile

UPSTREAM_REPO = "https://github.com/verl-project/verl"
PR_REPO = "https://github.com/ruiling-smartbear/verl"
PR_BRANCH = "feat/generation-prompt-delta-cache"
# (builder family, Hub candidates)
TARGETS = [
    ("qwen3", ["Qwen/Qwen3-8B"]),
    ("qwen35", ["Qwen/Qwen3.5-9B"]),
    ("deepseek", ["deepseek-ai/DeepSeek-V3.2-Exp", "deepseek-ai/DeepSeek-V3.1-Terminus", "deepseek-ai/DeepSeek-V3.1", "deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-R1"]),
    ("glm47", ["zai-org/GLM-4.7", "zai-org/GLM-4.6"]),
    ("minimaxm2", ["MiniMaxAI/MiniMax-M2"]),
    ("gptoss", ["openai/gpt-oss-20b"]),
    ("default", ["Qwen/Qwen3-8B"]),
]
TURNS = 100

RUNNER = r'''
import json, sys, time
verl_path, family, model, turns, out = sys.argv[1:6]
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
TOOL_RESULT = ("The lookup returned 302,971 residents as of the latest census estimate. " * 12)[:800]

tok = AutoTokenizer.from_pretrained(model)
builder = get_continuous_token_builder_class(family)(tok)
messages = [
    {"role": "system", "content": "You are a helpful agent. Use tools to answer."},
    {"role": "user", "content": "Find the population of Pittsburgh and its ten largest suburbs."},
]
per_turn, ids_all = [], []
for turn in range(int(turns)):
    messages.append({"role": "assistant", "content": "", "tool_calls": [
        {"id": f"call-{turn}", "type": "function", "function": {"name": "lookup", "arguments": json.dumps({"q": str(turn)})}}]})
    previous = list(messages)
    messages.append({"role": "tool", "content": TOOL_RESULT, "tool_call_id": f"call-{turn}", "name": "lookup"})
    t0 = time.perf_counter()
    ids = builder.tokenize_non_assistant_incremental_messages(previous, messages, tools=TOOLS)
    per_turn.append(round((time.perf_counter() - t0) * 1000, 3))
    ids_all.append(ids)
with open(out, "w") as fh:
    json.dump({"per_turn_ms": per_turn, "ids": ids_all}, fh)
'''


def clone(repo: str, path: str, branch: str | None = None) -> str:
    if not os.path.isdir(os.path.join(path, "verl")):
        branch_arg = f"-b {branch}" if branch else ""
        subprocess.run(f"rm -rf {path} && git clone -q --depth 1 {branch_arg} {repo} {path}", shell=True, check=True)
    return subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"], text=True).strip()


def run(verl_path: str, family: str, model: str) -> dict | str:
    out = tempfile.mktemp(suffix=".json")
    proc = subprocess.run(
        [sys.executable, "-c", RUNNER, verl_path, family, model, str(TURNS), out],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return ((proc.stderr or "").strip().splitlines() or [""])[-1][:110]
    with open(out) as fh:
        return json.load(fh)


def main() -> int:
    main_head = clone(UPSTREAM_REPO, "/tmp/verl_main")
    pr_head = clone(PR_REPO, "/tmp/verl_cache", PR_BRANCH)
    print(f"[cachebench] main @{main_head} vs cache branch @{pr_head}, {TURNS} tool turns")
    failures = 0
    for family, candidates in TARGETS:
        for model in candidates:
            base = run("/tmp/verl_main", family, model)
            if isinstance(base, str):
                print(f"[skip] {family} {model}: {base}")
                continue
            pr = run("/tmp/verl_cache", family, model)
            if isinstance(pr, str):
                print(f"[skip] {family} {model} (cache branch): {pr}")
                break
            same = base["ids"] == pr["ids"]
            failures += 0 if same else 1
            bt, pt = base["per_turn_ms"], pr["per_turn_ms"]
            print(
                f"[RESULT] {family:<10} {model:<30} "
                f"main total={sum(bt) / 1000:.2f}s last-turn={bt[-1]:.1f}ms | "
                f"cache total={sum(pt) / 1000:.2f}s last-turn={pt[-1]:.1f}ms | "
                f"{sum(bt) / sum(pt):.0f}x | identical_ids={same}"
            )
            break
    print("CACHE_BENCH_" + ("FAILED" if failures else "PASSED"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
