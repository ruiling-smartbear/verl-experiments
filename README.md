# Experiments behind my verl contributions

Scripts, raw outputs and write-ups for the verl issues and PRs listed below, so anyone reviewing them can rerun the numbers: the continuous-token generation-prompt experiments (#7617, #7628), the DeepSeek tool-append fix (#7630) and its follow-ups (#7635, #7636). Everything ran on tokenizer files only unless marked GPU; each script clones the verl commits it compares, so it is reproducible on its own.

Older links in those threads point at the `bench/975` branch of my sglang-omni fork, where this material first lived; that branch is kept unchanged so the links keep working.

## verl #7617 — generation-prompt delta in incremental tokenization

| question | script | write-up / where the numbers went |
|---|---|---|
| Can the generation prompt come out of the **last append group's render** (`add_generation_prompt=True` on that render) instead of two full-history renders? | [`verl_fold_probe_harness.py`](verl_fold_probe_harness.py) (trajectories shaped like `tool_agent_loop`'s, 24 cases per model, Gemma 4 included, full-history reference column); [`verl_fold_probe.py`](verl_fold_probe.py) is the earlier API-shaped matrix | [`generation_prompt_fold_experiment.md`](generation_prompt_fold_experiment.md) — 192 template-rendered cases over 8 rows (7 distinct templates), 0 differences; posted on [#7617](https://github.com/verl-project/verl/issues/7617#issuecomment-5473497404) |
| After the merge: does the #7628 merge commit (3dab856) reproduce its parent's (8e4a572) token ids, and what happens on the DeepSeek family (V3, R1, V3.1, V3.2-Exp)? | [`verl_fused_check.py`](verl_fused_check.py) | [`generation_prompt_fold_experiment.md#after-the-merge`](generation_prompt_fold_experiment.md#after-the-merge--7628-vs-its-parent), raw output in [`verl_fused_check_results.txt`](verl_fused_check_results.txt); posted on [#7628](https://github.com/verl-project/verl/pull/7628) |
| Does the DeepSeek fix (synthetic tool-call arguments as a JSON string) render tool appends on the real V3-family tokenizers and leave every other builder untouched? | [`verl_deepseek_fix_check.py`](verl_deepseek_fix_check.py) | [`generation_prompt_fold_experiment.md#the-fix-checked`](generation_prompt_fold_experiment.md#the-fix-checked), raw output in [`verl_deepseek_fix_check_results.txt`](verl_deepseek_fix_check_results.txt); opened as [#7630](https://github.com/verl-project/verl/pull/7630) |
| Follow-ups: DeepSeek V3-family tool parser + tool outputs after the first tool turn — tokenizer-level check of the prefix change against main | [`verl_deepseek_followup_check.py`](verl_deepseek_followup_check.py), patches in [`verl/patches/`](patches) | [`generation_prompt_fold_experiment.md#follow-ups-after-7630--7635-and-7636`](generation_prompt_fold_experiment.md#follow-ups-after-7630--7635-and-7636), raw output in [`verl_deepseek_followup_check_results.txt`](verl_deepseek_followup_check_results.txt) |
| Rollout-level check on a real model (DeepSeek-R1-0528-Qwen3-8B, sglang, 1× H100): before #7630 / main / main + patches | [`verl_deepseek_e2e.py`](verl_deepseek_e2e.py) | same section; raw output in [`verl_deepseek_e2e_results.txt`](verl_deepseek_e2e_results.txt) |
| Which builder does verl pick for the DeepSeek-R1 distills, does it construct, and which builder renders their tool appends like the template? | [`verl_distill_builder_probe.py`](verl_distill_builder_probe.py) | raw output in [`verl_distill_builder_probe_results.txt`](verl_distill_builder_probe_results.txt): since #6804 the Qwen builder is picked by `model_type` and fails to construct (no `<\|im_end\|>`); the DeepSeek builder, or the default builder with string synthetic arguments, renders them |
| The two #7637 candidate fixes (default-builder fallback vs tokenizer special-token key), each applied alone on `main`: identity on 13 models, factory selection, template-reference renders, special-token key inputs | [`verl_distill_fix_check.py`](verl_distill_fix_check.py), patches in [`patches/`](patches) | raw output in [`verl_distill_fix_check_results.txt`](verl_distill_fix_check_results.txt); the two PRs on [#7637](https://github.com/verl-project/verl/issues/7637) |
| Option 3 of #7637 as a PR: `data.continuous_token.model_family` override + an actionable construction error — identity on 13 models, auto error text, explicit-selection renders | [`verl_family_config_check.py`](verl_family_config_check.py), patch in [`patches/`](patches) | raw output in [`verl_family_config_check_results.txt`](verl_family_config_check_results.txt); opened as [#7640](https://github.com/verl-project/verl/pull/7640) |
| Is the delta **constant across turns** per family, and does it depend on the final role? | [`verl_cacheability_check.py`](verl_cacheability_check.py) | table below; the role dependence (DeepSeek emits nothing after a tool output) is posted on [#7617](https://github.com/verl-project/verl/issues/7617#issuecomment-5473506807) |
| Does **caching** the delta (keyed by final role + tools, revalidated) reproduce main's token ids, and how much does it save? | [`verl_cache_bench.py`](verl_cache_bench.py) | table in the [#7619](https://github.com/verl-project/verl/pull/7619) description |
| What does each template's `add_generation_prompt` guard actually read? | [`verl_template_snippets.py`](verl_template_snippets.py) | Jinja table in [`generation_prompt_experiment.md`](generation_prompt_experiment.md) |
| Where exactly does DeepSeek-V3.2-Exp's template fail on verl's synthetic tool prefix? | [`verl_deepseek_template.py`](verl_deepseek_template.py) | `arguments: {}` (a mapping) concatenated as a string — `TypeError: can only concatenate str (not "dict")` |
| Why does the Gemma 4 tokenizer not load, and what does its generation-prompt guard read? | [`verl_gemma4_load.py`](verl_gemma4_load.py) | needs transformers ≥ 5; the guard reads `ns.prev_message_type` (nothing after a tool response) |
| Why do the three gpt-oss renders disagree on a tool-final append? | [`verl_gptoss_detail.py`](verl_gptoss_detail.py) | decoded diff in both write-ups: the tool path is hand-built, and tool content is not JSON-encoded |
| Earlier: bounded pseudo-tail render vs full-history render, 28 scenarios × 4 Qwen models, 50/100/200-turn timings | [`verl_template_check.py`](verl_template_check.py) | superseded by the cache design; numbers were in the first version of the #7619 description |

### Delta constant across 20 tool turns? ([`verl_cacheability_check.py`](verl_cacheability_check.py), main @a0bd149)

| family | model | constant over 20 turns | same for tool / user / system final |
|---|---|---|---|
| qwen25 | Qwen2.5-7B-Instruct | yes | yes |
| qwen3 | Qwen3-8B | yes | yes |
| qwen35 | Qwen3.5-9B | yes | no — the template refuses a non-leading system message |
| minimaxm2 | MiniMax-M2 | yes | yes |
| glm47 | GLM-4.7 | yes | yes |
| gptoss | gpt-oss-20b | yes | yes |
| deepseek | DeepSeek-V3.2-Exp | yes per role | **no** — `<｜Assistant｜></think>` after a user turn, empty after a tool output |
| default | Qwen3-8B via the base builder | yes | yes |

Not loadable: GLM-5, Gemma-4/3 (gated), DeepSeek-V4 (gated); VL builders need a processor.

### Cache vs main, 100-turn tool loops ([`verl_cache_bench.py`](verl_cache_bench.py))

| builder | model | main | cache branch | last turn | ids |
|---|---|---|---|---|---|
| qwen3 | Qwen3-8B | 6.31s | 0.19s | 147 ms → 1.5 ms | identical |
| qwen35 | Qwen3.5-9B | 6.16s | 0.25s | 117 ms → 1.9 ms | identical |
| glm47 | GLM-4.7 | 4.50s | 0.30s | 91 ms → 1.4 ms | identical |
| minimaxm2 | MiniMax-M2 | 4.94s | 0.31s | 98 ms → 1.4 ms | identical |
| gptoss | gpt-oss-20b | 5.55s | 0.22s | 111 ms → 0.3 ms | identical |
| deepseek | DeepSeek-V3 | 4.80s | 0.22s | 96 ms → 0.7 ms | identical |
| default | Qwen3-8B | 6.14s | 0.35s | 112 ms → 1.6 ms | identical |

