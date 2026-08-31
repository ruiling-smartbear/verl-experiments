> **Superseded.** This was the first pass and two of its conclusions were too strong:
> DeepSeek-V3.2-Exp and gpt-oss are *not* counter-examples to rendering the generation
> prompt with the last append group — DeepSeek's role-dependent prompt is reproduced by
> that render (the dependence only rules out one cached id per model), and the gpt-oss
> difference is its hand-built tool path, not the template. The measurement that answers
> the question as asked is [`generation_prompt_fold_experiment.md`](generation_prompt_fold_experiment.md).
> The template-reading table and the gpt-oss decoded diff below are still accurate.

# Does the generation prompt need the full history? — experiment for verl #7617

**Question.** In `tokenize_non_assistant_incremental_messages`, the tokens added by
`add_generation_prompt=True` are produced today by a separate full-history render.
Is that necessary, or can the generation prompt be folded into the encode of the
last append group, by simply passing `add_generation_prompt=True` on that render?

**Answer, short.** For 7 of the 9 chat templates I could load, the generation
prompt is a constant literal emitted after the message loop and folding is exact.
Two templates are not safe to fold blindly: **gpt-oss**, where `add_generation_prompt`
also changes how the *last assistant turn* is rendered, and **DeepSeek-V3.2-Exp**,
where the generation prompt is computed from namespace state accumulated over the
whole history and is **empty** when the conversation ends on a tool output.

Setup: upstream `main` @8e4a572, real tokenizers and chat templates from the Hub
(tokenizer files only, no weights), CPU. Scripts:
[`verl_generation_prompt_probe.py`](verl_generation_prompt_probe.py),
[`verl_template_snippets.py`](verl_template_snippets.py),
[`verl_gptoss_detail.py`](verl_gptoss_detail.py).

## The conversation used

Built per case as `sys + user + (assistant tool call + tool response) * prior_turns`
followed by one append group. `prior_turns` is 0 and 10; the appended group is a
tool response, a user message or a system message; tools are passed and not passed.
The tool-final variant (`prior_turns=0`), which is the one the issue is about:

```json
{
  "previous": [
    {"role": "system", "content": "You are a helpful agent. Use tools to answer."},
    {"role": "user", "content": "Find the population of Pittsburgh."},
    {"role": "assistant", "content": "", "tool_calls": [
      {"id": "call-final", "type": "function",
       "function": {"name": "lookup", "arguments": {"q": "99"}}}]}
  ],
  "appended (incremental)": [
    {"role": "tool", "name": "lookup", "tool_call_id": "call-final",
     "content": "Pittsburgh had 302,971 residents at the latest census estimate."}
  ],
  "tools": [
    {"type": "function", "function": {"name": "lookup",
      "description": "Look up a population figure.",
      "parameters": {"type": "object", "properties": {"q": {"type": "string"}},
                     "required": ["q"]}}}
  ]
}
```

Three quantities are compared per case:

| name | how it is produced |
|---|---|
| `truth` | `render_delta_token_id(previous, appended, add_generation_prompt=True)` — render the whole conversation twice and take the suffix |
| `current` | what `main` does: bounded render of the append group, plus `_tokenize_generation_prompt_delta` (a full-history render) |
| `folded` | the proposal: render the append group once with `add_generation_prompt=True`, nothing else |

## Method 1 — reading the templates

The Jinja guarded by `add_generation_prompt`, and whether it reads anything from
the message history:

| family | model | generation prompt block | reads history? |
|---|---|---|---|
| qwen | Qwen2-7B-Instruct | `{% if add_generation_prompt %}{{ '<\|im_start\|>assistant\n' }}{% endif %}` | no |
| qwen25 | Qwen2.5-7B-Instruct | same literal | no |
| qwen3 | Qwen3-8B | same literal, plus `<think>\n\n</think>` when `enable_thinking` is false | no |
| qwen35 | Qwen3.5-9B | same literal, plus a `<think>` variant on the kwarg | no |
| minimaxm2 | MiniMax-M2 | `{%- if add_generation_prompt -%}{{- ']~b]ai' ~ '\n' ~ '<think>' ~ '\n' }}` | no |
| glm47 | GLM-4.7 | `{%- if add_generation_prompt -%}<\|assistant\|>{{ '</think>' if not enable_thinking else '<think>' }}` | no |
| default | SmolLM3-3B | `<\|im_start\|>assistant\n` (+ empty think block by reasoning mode) | no |
| **gptoss** | gpt-oss-20b | tail guard is the literal `<\|start\|>assistant`, **but** `add_generation_prompt` is also read inside the message loop: `{%- elif loop.last and not add_generation_prompt %}` (renders the last assistant turn's CoT only when the flag is false) | **yes, inside the loop** |
| **deepseek** | DeepSeek-V3.2-Exp | `{% if add_generation_prompt and not ns.is_tool %}{% if ns.is_last_user or ns.is_only_sys or not ns.is_prefix %}{{'<｜Assistant｜>'}}...` | **yes, namespace state over the whole history** |

Not loadable here: GLM-5 (`TokenizersBackend` not in this transformers), Gemma-4/Gemma-3
(gated), DeepSeek-V4 (gated), MiniMax-Text-01 (its template rejects this message shape).
The VL builders need a processor and were not covered.

## Method 2 — measuring

12 cases per model (tools on/off x prior_turns 0/10 x append role tool/user/system):

| family | model | `folded == current` | `current == truth` | decoded generation prompt |
|---|---|---|---|---|
| qwen | Qwen2-7B-Instruct | 12/12 | 12/12 | `<\|im_start\|>assistant\n` |
| qwen25 | Qwen2.5-7B-Instruct | 12/12 | 12/12 | `<\|im_start\|>assistant\n` |
| qwen3 | Qwen3-8B | 12/12 | n/a (see below) | `<\|im_start\|>assistant\n` |
| qwen35 | Qwen3.5-9B | 8/8 rendered (4 refused by the template) | 4/4 rendered | `<\|im_start\|>assistant\n<think>\n` |
| minimaxm2 | MiniMax-M2 | 12/12 | 12/12 | `]~b]ai\n<think>\n` |
| glm47 | GLM-4.7 | 12/12 | 12/12 | `<\|assistant\|><think>` |
| default | Qwen3-8B, SmolLM3-3B | 24/24 | 12/12 rendered | `<\|im_start\|>assistant\n` |
| **gptoss** | gpt-oss-20b | 8/12 (all four tool-final cases differ) | 0/4 rendered | `<\|start\|>assistant` |
| **deepseek** | DeepSeek-V3.2-Exp | 4/4 rendered (user-final); tool-final and system-final do not render | 4/4 rendered | `<｜Assistant｜></think>` after a user turn, **empty** after a tool output |

"n/a" and "not rendered" are cases where a path raises rather than disagrees; they
are listed under known limitations below.

### gpt-oss, tool-final: the one real disagreement

```
truth   : '<|start|>functions.lookup to=assistant<|channel|>commentary<|message|>"Pittsburgh had 302,971 residents at the latest census estimate."<|end|><|start|>assistant'
current : '<|start|>functions.lookup to=assistant<|channel|>commentary<|message|>Pittsburgh had 302,971 residents at the latest census estimate.<|end|><|start|>assistant'
folded  : '<|start|>functions.lookup to=assistant<|channel|>commentary<|message|>Pittsburgh had 302,971 residents at the latest census estimate.<|end|>'
```

Two separate things:

1. `folded` loses the generation prompt because `GptOssContinuousTokenBuilder._tokenize_tool_group`
   does not use the chat template at all — it formats the tool response with an
   f-string and calls `tokenizer.encode`. Passing `add_generation_prompt=True` has
   nothing to act on. Folding for gpt-oss means appending the constant
   `<|start|>assistant` to that hand-built string, which is exact and trivial.
2. `current != truth` here is unrelated to this issue and looks like a pre-existing
   gap: the template writes tool content JSON-encoded (`"..."`), while
   `_format_tool_response` writes it raw. Every gpt-oss tool turn produced by the
   incremental path therefore differs from what the template would have produced.

### DeepSeek: the generation prompt is not a constant

The guard is `{% if add_generation_prompt and not ns.is_tool %}`, so after a tool
output DeepSeek emits **no generation prompt at all**, and the `ns.is_prefix` /
`ns.is_only_sys` conditions read the whole history. Measured: `<｜Assistant｜></think>`
after a user turn, empty string after a tool output. Any scheme that assumes one
constant generation prompt per model is wrong here; a scheme keyed on the final
message role gets this case right, and one that re-validates would catch the rest.

## Known limitations hit during the run

- **The naive full-history suffix diff is not always available.** For Qwen3-8B
  (all three append roles) and gpt-oss (user/system appends), rendering
  `previous` is not a token prefix of rendering `previous + appended`, so `truth`
  raises `Continuous Token token-id suffix diff failed`. This is exactly why the
  builders render bounded pseudo conversations instead; it also means "compare to
  the full render" is not a universally available oracle.
- **Qwen3.5 refuses a system message that is not first** (`TemplateError: System
  message must be at the beginning.`) although `_SUPPORTED_APPEND_ROLES` allows
  appending `system`. Same error on main and on any variant.
- **Tool-call argument encoding is not uniform**: DeepSeek's template requires
  `function.arguments` as a JSON string, while Qwen / GLM / MiniMax require a
  mapping. The probe picks whichever renders.

## What this suggests

1. For the ChatML-style families (Qwen 2/2.5/3/3.5, MiniMax-M2, GLM-4.7, SmolLM3
   and anything else whose guard is a literal), the generation prompt can be folded
   into the last append group's render. That removes the extra render entirely —
   no cache, no re-validation, no full history — and the measurements agree
   token-for-token with what `main` produces today.
2. gpt-oss needs one line rather than a mechanism: its tool group is hand-built, so
   append the constant generation prompt there.
3. DeepSeek is the counter-example that says this cannot be a blanket default: its
   generation prompt depends on accumulated state and is empty after tool outputs.
   Keying on the final message role reproduces it correctly in the cases that
   render, but the safe shape is still "fold where the template is verified, keep
   the full render as the fallback elsewhere".
