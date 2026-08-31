# Real-tokenizer check + benchmark for verl #7617 / PR #7619.
#
# Runs the same inputs through two checkouts of verl, upstream main and the PR
# branch, with real Qwen tokenizers and chat templates:
#   1. scenario matrix: generation-prompt delta for user/tool/multi-tool/
#      assistant-then-user/long-tool-loop histories, tools on/off, thinking on/off
#   2. tool loop: a multi-turn agent rollout driven through the public
#      tokenize_non_assistant_incremental_messages API, timed per turn
# Token ids from both checkouts are compared, so the timing run doubles as a
# correctness check. Tokenization only, CPU only.
import json
import os
import subprocess
import sys
import tempfile

UPSTREAM_REPO = "https://github.com/verl-project/verl"
PR_REPO = "https://github.com/ruiling-smartbear/verl"
PR_BRANCH = "fix/bounded-generation-prompt-delta"
MODELS = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-8B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-35B-A3B"]
BENCH_MODELS = ["Qwen/Qwen3-8B", "Qwen/Qwen3.5-9B"]
TURNS = "50,100,200"

RUNNER = r'''
import json, sys, time
from typing import Any
verl_path, model, turns_csv, out = sys.argv[1:5]
sys.path.insert(0, verl_path)
from transformers import AutoTokenizer
from verl.utils.tokenizer.continuous_token import QwenContinuousTokenBuilder

TOOLS = [{
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Look up a population figure.",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
    },
}]
TOOL_RESULT = ("The lookup returned 302,971 residents as of the latest census estimate. " * 12)[:800]


def scenarios() -> list[tuple[str, list[dict[str, Any]]]]:
    tool_call = {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": {}}}
    base = [
        {"role": "system", "content": "You are a helpful agent."},
        {"role": "user", "content": "Find the population of Pittsburgh."},
    ]
    asst = {"role": "assistant", "content": "", "tool_calls": [tool_call]}
    tool = {"role": "tool", "content": "302,971", "tool_call_id": "call-1", "name": "lookup"}
    long_tail = []
    for turn in range(12):
        long_tail.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{turn}", "type": "function", "function": {"name": "lookup", "arguments": {}}}]})
        long_tail.append({"role": "tool", "content": f"partial {turn}", "tool_call_id": f"c{turn}"})
    second_call = {"id": "call-2", "type": "function", "function": {"name": "lookup", "arguments": {}}}
    return [
        ("user-last", base),
        ("tool-last", base + [asst, tool]),
        ("system-last", base + [asst, tool, {"role": "system", "content": "Budget: three more tool calls."}]),
        ("tool-then-user", base + [asst, tool, {"role": "user", "content": "Thanks, keep going."}]),
        ("multi-tool-last", base + [{**asst, "tool_calls": [tool_call, second_call]}, tool, {**tool, "tool_call_id": "call-2"}]),
        ("assistant-then-user", base + [{"role": "assistant", "content": "Let me check.", "reasoning_content": "thinking..."}, {"role": "user", "content": "Please do."}]),
        ("long-tool-loop", base + long_tail),
    ]


def tool_loop(builder, turns):
    messages = [
        {"role": "system", "content": "You are a helpful agent. Use tools to answer."},
        {"role": "user", "content": "Find the population of Pittsburgh and its ten largest suburbs."},
    ]
    per_turn, ids_all = [], []
    for turn in range(turns):
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"call-{turn}", "type": "function", "function": {"name": "lookup", "arguments": {"q": str(turn)}}}]})
        previous = list(messages)
        messages.append({"role": "tool", "content": TOOL_RESULT, "tool_call_id": f"call-{turn}", "name": "lookup"})
        t0 = time.perf_counter()
        ids = builder.tokenize_non_assistant_incremental_messages(previous, messages, tools=TOOLS)
        per_turn.append(time.perf_counter() - t0)
        ids_all.append(ids)
    return per_turn, ids_all


tok = AutoTokenizer.from_pretrained(model)
result = {"matrix": {}, "bench": {}}
for kwargs in ({}, {"enable_thinking": False}):
    builder = QwenContinuousTokenBuilder(tok, chat_template_kwargs=dict(kwargs))
    for name, conversation in scenarios():
        for tools in (None, TOOLS):
            key = f"kwargs={kwargs} scenario={name} tools={bool(tools)}"
            try:
                result["matrix"][key] = builder._tokenize_generation_prompt_delta(conversation, tools=tools)
            except Exception as exc:  # noqa: BLE001
                result["matrix"][key] = f"ERROR {type(exc).__name__}: {str(exc)[:110]}"
if turns_csv:
    builder = QwenContinuousTokenBuilder(tok)
    tool_loop(builder, 3)  # warm-up: template compile, tokenizer caches
    for turns in [int(t) for t in turns_csv.split(",")]:
        per_turn, ids_all = tool_loop(builder, turns)
        result["bench"][str(turns)] = {"per_turn_ms": [round(t * 1000, 3) for t in per_turn], "ids": ids_all}
with open(out, "w") as fh:
    json.dump(result, fh)
'''


def clone(repo: str, path: str, branch: str | None = None) -> str:
    if not os.path.isdir(os.path.join(path, "verl")):
        branch_arg = f"-b {branch}" if branch else ""
        subprocess.run(f"rm -rf {path} && git clone -q --depth 1 {branch_arg} {repo} {path}", shell=True, check=True)
    return subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"], text=True).strip()


def run_checkout(verl_path: str, model: str, turns: str) -> dict:
    out = tempfile.mktemp(suffix=".json")
    subprocess.run([sys.executable, "-c", RUNNER, verl_path, model, turns, out], check=True)
    with open(out) as fh:
        return json.load(fh)


def main() -> int:
    main_head = clone(UPSTREAM_REPO, "/tmp/verl_main")
    pr_head = clone(PR_REPO, "/tmp/verl_pr", PR_BRANCH)
    print(f"[bench] upstream main @{main_head} vs PR branch @{pr_head}")
    failures = 0
    for model in MODELS:
        turns = TURNS if model in BENCH_MODELS else ""
        try:
            base = run_checkout("/tmp/verl_main", model, turns)
            pr = run_checkout("/tmp/verl_pr", model, turns)
        except subprocess.CalledProcessError as exc:
            failures += 1
            print(f"[ERROR] {model}: runner exited {exc.returncode}")
            continue
        matrix_ok = 0
        same_error = 0
        for key, want in base["matrix"].items():
            got = pr["matrix"].get(key)
            if got != want:
                failures += 1
                print(f"[MISMATCH] {model} {key} main={want} pr={got}")
            elif str(want).startswith("ERROR"):
                # The template refuses this history on both sides, identically:
                # same behaviour, so it is not a regression.
                same_error += 1
            else:
                matrix_ok += 1
        note = f" (+{same_error} refused by the template on both sides)" if same_error else ""
        print(
            f"[CHECK] {model} generation-prompt delta identical in "
            f"{matrix_ok}/{len(base['matrix']) - same_error} rendered scenarios{note}"
        )
        for turns_key, b in base["bench"].items():
            p = pr["bench"][turns_key]
            same = b["ids"] == p["ids"]
            if not same:
                failures += 1
            bt, pt = b["per_turn_ms"], p["per_turn_ms"]
            print(
                f"[RESULT] {model} turns={turns_key} "
                f"main total={sum(bt) / 1000:.2f}s per-turn {bt[0]:.1f}->{bt[-1]:.1f}ms | "
                f"pr total={sum(pt) / 1000:.2f}s per-turn {pt[0]:.1f}->{pt[-1]:.1f}ms | "
                f"{sum(bt) / sum(pt):.0f}x | identical_ids={same}"
            )
    print("TEMPLATE_CHECK_" + ("FAILED" if failures else "PASSED"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
