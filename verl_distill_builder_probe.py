# Which Continuous Token builder does verl main pick for the DeepSeek-R1 distills,
# does it construct, and which builder renders their tool appends like the template?
#
# Tokenizer files only. Only verl's tokenizer package is imported (parent packages
# stubbed), so a bare venv with transformers + jinja2 is enough.
import json
import os
import subprocess
import sys
import types

VERL = "/tmp/verl_now"
if not os.path.isdir(os.path.join(VERL, "verl")):
    subprocess.run(f"git clone -q --depth 1 https://github.com/verl-project/verl {VERL}", shell=True, check=True)
print("verl main:", subprocess.check_output(["git", "-C", VERL, "log", "--oneline", "-1"], text=True).strip())
for name, sub in (("verl", "verl"), ("verl.utils", "verl/utils")):
    stub = types.ModuleType(name)
    stub.__path__ = [f"{VERL}/{sub}"]
    sys.modules[name] = stub

import transformers  # noqa: E402
from transformers import AutoConfig, AutoTokenizer  # noqa: E402
from verl.utils.tokenizer import continuous_token as ct  # noqa: E402
from verl.utils.tokenizer.continuous_token_wiring import (  # noqa: E402
    create_continuous_token_builder,
    get_continuous_token_builder_class,
    infer_continuous_token_model_family,
)

# (model, tokenizer class to load with; DeepSeek-R1-0528-Qwen3-8B decodes wrongly as
#  LlamaTokenizerFast under transformers 5, so load it as the Qwen2 tokenizer it is)
MODELS = [
    ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "Qwen2TokenizerFast"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", ""),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", ""),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", ""),
]
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
    out = []
    for message in messages:
        message = dict(message)
        if message.get("tool_calls"):
            message["tool_calls"] = [
                {**call, "function": {**call["function"], "arguments": json.dumps(call["function"]["arguments"])}}
                for call in message["tool_calls"]
            ]
        out.append(message)
    return out


original_synthetic = ct.ContinuousTokenBuilder._synthetic_assistant_for_tools


def string_synthetic(self, tool_messages):
    message = original_synthetic(self, tool_messages)
    for call in message["tool_calls"]:
        call["function"]["arguments"] = "{}"
    return message


for model, tokenizer_class in MODELS:
    loader = getattr(transformers, tokenizer_class) if tokenizer_class else AutoTokenizer
    tok = loader.from_pretrained(model)
    model_type = AutoConfig.from_pretrained(model).model_type
    family = infer_continuous_token_model_family(hf_model_type=model_type)
    print(f"===== {model} | model_type={model_type} | verl picks family={family.value}")
    print(f"   <|im_end|> id: {tok.convert_tokens_to_ids('<|im_end|>')} | eos_token: {tok.eos_token!r}")
    try:
        picked = create_continuous_token_builder(tok, hf_model_type=model_type)
        print(f"   construction: ok ({type(picked).__name__})")
    except Exception as exc:  # noqa: BLE001
        print(f"   construction FAILS: {type(exc).__name__}: {str(exc)[:90]}")
    for turns in (1, 3):
        previous = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
        for i in range(turns - 1):
            previous += [turn(i), tool(i)]
        previous.append(turn(turns - 1))
        appended = [tool(turns - 1)]
        full = tok.apply_chat_template(with_string_arguments(previous + appended), tokenize=True,
                                       add_generation_prompt=True, return_dict=False)
        prefix = tok.apply_chat_template(with_string_arguments(previous), tokenize=True,
                                         add_generation_prompt=False, return_dict=False)
        reference = full[len(prefix):] if full[: len(prefix)] == prefix else None
        candidates = [
            ("default builder, main", ct.ContinuousTokenBuilder, None),
            ("default builder + string synthetic arguments", ct.ContinuousTokenBuilder, string_synthetic),
            ("deepseek builder, main", get_continuous_token_builder_class("deepseek"), None),
        ]
        for label, builder_cls, patch in candidates:
            ct.ContinuousTokenBuilder._synthetic_assistant_for_tools = patch or original_synthetic
            try:
                ids = builder_cls(tok).tokenize_non_assistant_incremental_messages(previous, previous + appended, tools=TOOLS)
                print(f"   tool turn {turns} [{label}]: {tok.decode(ids)!r} | == template: {ids == reference}")
            except Exception as exc:  # noqa: BLE001
                print(f"   tool turn {turns} [{label}]: FAILS {type(exc).__name__}: {str(exc)[:70]}")
        ct.ContinuousTokenBuilder._synthetic_assistant_for_tools = original_synthetic
print("DISTILL_PROBE_DONE")
