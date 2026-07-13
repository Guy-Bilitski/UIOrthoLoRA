"""Ensure the installed lm-eval `bbh_fewshot` (non-CoT) metric normalizes case and
strips surrounding whitespace / trailing '.' before exact_match.

WHY: bbh_fewshot scores raw exact_match with NO filter. A base model like Qwen2.5
emits the answer with a leading space (" -33" for target "-33"), so *correct* answers
were scored wrong -> bbh=0.00 for Qwen while Llama-2 (no leading space) was unaffected.
Fix = lower-case + strip leading/trailing whitespace and trailing periods. Deliberately
NOT ignore_punctuation (that would delete minus signs and corrupt numeric answers).

Validated: Qwen bbh 0.00 -> 0.54; Llama-2 0.47 -> 0.47 (byte-identical no-op, every
subtask) so the existing Llama-2 cells stay valid. Idempotent; safe to call every run.
See handoff/16_BBH_METRIC_FIX_2026-07-01.md.
"""
import os
import lm_eval

_OLD = ("    higher_is_better: true\n"
        "    # ignore_case: true\n"
        "    # ignore_punctuation: true")
_NEW = ("    higher_is_better: true\n"
        "    ignore_case: true\n"
        "    # NOT ignore_punctuation (would strip minus signs). Strip surrounding\n"
        "    # whitespace + trailing '.' so a base model's ' -33'/'24.' matches '-33'/'24'.\n"
        "    regexes_to_ignore:\n"
        '      - "^\\\\s+"\n'
        '      - "[\\\\s.]+$"')


def ensure_bbh_fewshot_metric_fix(verbose=True):
    p = os.path.join(os.path.dirname(lm_eval.__file__),
                     "tasks", "bbh", "fewshot", "_fewshot_template_yaml")
    if not os.path.exists(p):
        return False
    txt = open(p).read()
    if "regexes_to_ignore" in txt:
        return True  # already patched
    if _OLD in txt:
        open(p, "w").write(txt.replace(_OLD, _NEW))
        if verbose:
            print(f"[bbh_metric_fix] applied normalization to {p}", flush=True)
        return True
    if verbose:
        print(f"[bbh_metric_fix] WARNING: could not patch {p} (unexpected content)", flush=True)
    return False


if __name__ == "__main__":
    ensure_bbh_fewshot_metric_fix()
