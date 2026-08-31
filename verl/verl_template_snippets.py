# Print the part of each chat template that is guarded by add_generation_prompt
# (method 1 of the #7617 question: read the Jinja instead of measuring it).
import re

from transformers import AutoTokenizer

MODELS = [
    ("qwen", "Qwen/Qwen2-7B-Instruct"),
    ("qwen25", "Qwen/Qwen2.5-7B-Instruct"),
    ("qwen3", "Qwen/Qwen3-8B"),
    ("qwen35", "Qwen/Qwen3.5-9B"),
    ("minimaxm2", "MiniMaxAI/MiniMax-M2"),
    ("glm47", "zai-org/GLM-4.7"),
    ("gptoss", "openai/gpt-oss-20b"),
    ("deepseek", "deepseek-ai/DeepSeek-V3.2-Exp"),
    ("default", "HuggingFaceTB/SmolLM3-3B"),
]

for family, model in MODELS:
    try:
        tok = AutoTokenizer.from_pretrained(model)
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] {family} {model}: {type(exc).__name__}")
        continue
    template = getattr(tok, "chat_template", None)
    if not isinstance(template, str):
        print(f"[skip] {family} {model}: chat_template is {type(template).__name__}")
        continue
    hits = [m.start() for m in re.finditer("add_generation_prompt", template)]
    print(f"[TPL] {family} {model}: {len(hits)} mentions of add_generation_prompt, template {len(template)} chars")
    for hit in hits:
        block = " ".join(template[max(0, hit - 150) : hit + 300].split())
        print(f"[JINJA] {family}: ...{block}...")
    tail = template[hits[-1] :] if hits else ""
    print(f"[REFS] {family}: after the last guard, references loop/messages/ns: {bool(re.search(r'(loop\.|messages\[|ns\.|namespace\()', tail[:400]))}")
print("SNIPPETS_DONE")
