# Rollout-level check of verl's DeepSeek continuous-token path on a real model.
#
# Two tool turns, driven the way tool_agent_loop drives the pieces:
#   build_initial_tokens -> model generates -> verl's tool parser (deepseek_v3,
#   when the checkout has it) -> tool runs -> merge_non_assistant_tokens (the
#   tokenize_non_assistant_incremental_messages path) -> model continues ...
# The model is served by an sglang server; the driver only needs transformers
# (+ ray/pydantic/regex for verl's parser module). Run against several verl
# checkouts to compare: before #7630 the first append raises; with #7630 the
# appends render; with the outputs-prefix fix the second append matches the
# template's full-conversation render on V3 / R1-style templates.
import argparse
import asyncio
import json
import re
import sys
import time
import types
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--verl", required=True, help="path to a verl checkout")
parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
parser.add_argument("--server", default="http://127.0.0.1:30000")
parser.add_argument("--label", default="")
parser.add_argument("--max-new-tokens", type=int, default=2500)
parser.add_argument("--nothink", action="store_true", help="prefill an empty think block on the first turn")
parser.add_argument("--tokenizer-class", default="", help="transformers tokenizer class to load instead of Auto")
args = parser.parse_args()

for name, sub in (
    ("verl", "verl"),
    ("verl.utils", "verl/utils"),
    ("verl.tools", "verl/tools"),
    ("verl.experimental", "verl/experimental"),
    ("verl.experimental.agent_loop", "verl/experimental/agent_loop"),
):
    stub = types.ModuleType(name)
    stub.__path__ = [f"{args.verl}/{sub}"]
    sys.modules[name] = stub
import transformers  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from verl.utils.tokenizer.continuous_token_wiring import get_continuous_token_builder_class  # noqa: E402

try:
    from verl.experimental.agent_loop.tool_parser import ToolParser  # noqa: E402
except Exception as exc:  # noqa: BLE001
    ToolParser = None
    parser_import_error = f"{type(exc).__name__}: {str(exc)[:80]}"

TOOL_NAME = "get_population"
TOOLS = [{
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Return the latest population estimate of a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}]
POPULATIONS = {"pittsburgh": "302,971", "cleveland": "362,656"}
CALL_FORMAT = (
    "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_population\n"
    "```json\n{\"city\": \"<city name>\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
)
SYSTEM_PROMPT = (
    "You can call one tool, get_population(city), which returns the latest population estimate of a city. "
    "You do not know any population figures yourself. To call the tool, end your reply with exactly this block "
    "and nothing after it:\n" + CALL_FORMAT + "\nCall the tool for one city at a time. "
    "When you have every figure you need, answer the user in one sentence."
)
USER_PROMPT = "What are the current populations of Pittsburgh and Cleveland? Use the tool, one city per call."
# Tolerant of tokenizers whose decode drops the ｜ / ▁ characters of the DeepSeek markers
# (DeepSeek-R1-0528-Qwen3-8B under transformers 5 does), so the fallback still sees the call.
CALL_RE = re.compile(
    r"<｜?tool▁?calls▁?begin｜?><｜?tool▁?call▁?begin｜?>function<｜?tool▁?sep｜?>(\w+)\s*```json\s*(.*?)\s*```<｜?tool▁?call▁?end｜?><｜?tool▁?calls▁?end｜?>",
    re.S,
)


def generate(token_ids, max_new_tokens):
    payload = {
        "input_ids": token_ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens, "skip_special_tokens": False},
    }
    request = urllib.request.Request(
        args.server + "/generate", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def run_tool(arguments):
    city = str(arguments.get("city", "")).strip().lower()
    figure = POPULATIONS.get(city)
    return f"{city.title()}: {figure} residents (2023 census estimate)." if figure else f"No figure for {city!r}."


def with_string_arguments(messages):
    rendered = []
    for message in messages:
        message = dict(message)
        if message.get("tool_calls"):
            message["tool_calls"] = [
                {**call, "function": {**call["function"], "arguments": json.dumps(call["function"]["arguments"])}}
                for call in message["tool_calls"]
            ]
        rendered.append(message)
    return rendered


tag = f"[{args.label}] " if args.label else ""
tokenizer_cls = getattr(transformers, args.tokenizer_class) if args.tokenizer_class else AutoTokenizer
tok = tokenizer_cls.from_pretrained(args.model)
builder = get_continuous_token_builder_class("deepseek")(tok)
eos_id = builder._eos_id
verl_parser = None
if ToolParser is not None:
    try:
        verl_parser = ToolParser.get_tool_parser("deepseek_v3", tok)
    except ValueError as exc:
        print(f"{tag}no deepseek_v3 parser in this checkout ({exc}); falling back to a regex")
else:
    print(f"{tag}could not import verl's tool_parser ({parser_import_error}); falling back to a regex")
print(f"{tag}verl={args.verl} model={args.model} builder={type(builder).__name__} "
      f"parser={type(verl_parser).__name__ if verl_parser else 'regex'} eos_id={eos_id}")


def parse_calls(text, ids):
    if verl_parser is not None:
        content, calls = asyncio.run(verl_parser.extract_tool_calls(ids, tools=None))
        parsed = []
        for call in calls:
            try:
                parsed.append((call.name, json.loads(call.arguments)))
            except json.JSONDecodeError:
                print(f"{tag}dropping a call with invalid JSON arguments: {call.arguments[:80]!r}")
        if parsed:
            return content, parsed
    matches = list(CALL_RE.finditer(text))
    if not matches:
        return text, []
    match = matches[-1]  # the model may quote the format from the system prompt while thinking
    if verl_parser is not None:
        print(f"{tag}note: verl's parser saw no call but the tolerant regex did (decode dropped the marker characters)")
    return text[: match.start()], [(match.group(1), json.loads(match.group(2)))]


messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": USER_PROMPT}]
runtime_ids = builder.build_initial_tokens(messages, tools=TOOLS)
if args.nothink:
    runtime_ids = runtime_ids + tok.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
print(f"{tag}prompt: {len(runtime_ids)} tokens, tail {tok.decode(runtime_ids)[-120:]!r}")

forced_turns = []
appended_texts = []
final_text = ""
for turn in (1, 2, 3):
    t0 = time.time()
    out = generate(runtime_ids, args.max_new_tokens)
    text = out["text"]
    ids = out.get("output_ids") or tok.encode(text, add_special_tokens=False)
    print(f"{tag}turn {turn}: {len(ids)} tokens in {time.time() - t0:.1f}s, finish={out['meta_info'].get('finish_reason')}, "
          f"ids_from={'server' if out.get('output_ids') else 're-encoded text'}")
    text = tok.decode(ids)  # decode with the driver's tokenizer, not the server's
    print(f"{tag}turn {turn} text tail: {text[-300:]!r}")
    content, calls = parse_calls(text, ids)
    if calls:
        print(f"{tag}turn {turn}: model called {calls}")
        assistant_ids = list(ids)
    elif turn < 3:
        city = ["Pittsburgh", "Cleveland"][turn - 1]
        forced_turns.append(turn)
        content = "<think>\nI need the tool for this.\n</think>\n\n"
        assistant_ids = tok.encode(content + CALL_FORMAT.replace("<city name>", city), add_special_tokens=False)
        calls = [(TOOL_NAME, {"city": city})]
        print(f"{tag}turn {turn}: model did not call the tool; using a hand-written call for {city}")
    else:
        final_text = text
        break
    if assistant_ids[-1] != eos_id:
        assistant_ids.append(eos_id)
    assistant_message = {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {"id": f"call_{turn}_{index}", "type": "function", "function": {"name": name, "arguments": arguments}}
            for index, (name, arguments) in enumerate(calls)
        ],
    }
    tool_messages = [
        {"role": "tool", "content": run_tool(arguments), "tool_call_id": f"call_{turn}_{index}"}
        for index, (_, arguments) in enumerate(calls)
    ]
    runtime_ids = builder.merge_assistant_tokens(runtime_ids, assistant_ids).token_ids
    previous = messages + [assistant_message]
    updated = previous + tool_messages
    try:
        merged = builder.merge_non_assistant_tokens(previous, updated, runtime_ids, tools=TOOLS)
    except Exception as exc:  # noqa: BLE001
        print(f"{tag}turn {turn}: TOOL APPEND FAILED: {type(exc).__name__}: {str(exc)[:160]}")
        print(f"{tag}RESULT: append_failed_at_turn={turn} forced_turns={forced_turns}")
        raise SystemExit(0)
    appended = merged.token_ids[len(runtime_ids):]
    appended_text = tok.decode(appended)
    appended_texts.append(appended_text)
    # Independent reference: the template's own render of the whole conversation, suffix-diffed.
    try:
        full = tok.apply_chat_template(
            with_string_arguments(updated), tokenize=True, add_generation_prompt=True, return_dict=False
        )
        prefix = tok.apply_chat_template(
            with_string_arguments(previous), tokenize=True, add_generation_prompt=False, return_dict=False
        )
        reference = "matches template" if full[len(prefix):] == appended and full[: len(prefix)] == prefix else (
            f"differs from template: {tok.decode(full[len(prefix):])[:120]!r}")
    except Exception as exc:  # noqa: BLE001
        reference = f"template render failed: {type(exc).__name__}"
    print(f"{tag}turn {turn}: tool append {len(appended)} tokens -> {appended_text!r} | {reference}")
    runtime_ids = merged.token_ids
    messages = updated

answer_ok = all(figure in final_text for figure in POPULATIONS.values())
print(f"{tag}final answer tail: {final_text[-400:]!r}")
print(f"{tag}RESULT: appends_ok={len(appended_texts)} forced_turns={forced_turns} "
      f"parser={'verl' if verl_parser else 'regex'} answer_has_both_figures={answer_ok}")
