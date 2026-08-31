# Why does the Gemma 4 tokenizer fail to load here, and what does its template look like?
import traceback

import transformers
from transformers import AutoTokenizer

print("[env] transformers", transformers.__version__)
for model in ("google/gemma-4-12b-it", "unsloth/gemma-4-12b-it", "google/gemma-4-27b-it"):
    try:
        tok = AutoTokenizer.from_pretrained(model)
    except Exception as exc:  # noqa: BLE001
        frames = traceback.extract_tb(exc.__traceback__)
        last = frames[-1]
        print(f"[load FAIL] {model}: {type(exc).__name__}: {str(exc)[:120]} at {last.filename.split('site-packages/')[-1]}:{last.lineno} in {last.name}")
        continue
    template = tok.chat_template
    print(f"[load ok] {model}: chat_template type={type(template).__name__}", end=" ")
    if isinstance(template, dict):
        print("keys=", list(template.keys())[:6])
        template = next(iter(template.values()))
    elif isinstance(template, list):
        print("list of", len(template), [t.get("name") for t in template if isinstance(t, dict)][:6])
        template = template[0]["template"] if isinstance(template[0], dict) else template[0]
    else:
        print("len=", len(template or ""))
    if isinstance(template, str):
        for key in ("add_generation_prompt", "tool_response", "start_of_turn", "role == 'tool'", 'role == "tool"'):
            hits = [m.start() for m in __import__("re").finditer(__import__("re").escape(key), template)]
            print(f"[TPL] {key!r}: {len(hits)} mention(s)")
        i = template.rfind("add_generation_prompt")
        print("[TPL tail]", " ".join(template[max(0, i - 200): i + 300].split())[:480])
        print("[TOKEN] <|tool_response> id:", tok.convert_tokens_to_ids("<|tool_response>"), "unk:", tok.unk_token_id)
print("GEMMA4_LOAD_DONE")
